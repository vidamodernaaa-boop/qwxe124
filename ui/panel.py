"""N-panel UI - a tabbed, guided workflow (ZWrap-style wizard).

One compact panel: an icon tab strip in workflow order (each tab gets a check
badge when its step is done), a progress bar, and ONE step's controls at a
time inside a card. Steps whose prerequisites are missing show a single
locked line with a jump button instead of dead controls, so the panel never
grows long. Secondary settings sit in dark disclosure boxes (triangle to
open); explanations live in `?` popup dialogs, not inline labels.
"""

import os
import textwrap

import bpy

from ..core import landmarks as lmdata
from ..core import rest_tuning
from . import flow, icons
from .flow import MIN_PAIRS, RECOMMENDED_PAIRS

_WRAP_CHARS = 44  # popup text wrap width


# -------------------------------------------------------------- operators ---

class MHFRT_OT_set_tab(bpy.types.Operator):
    """Jump to a workflow step."""
    bl_idname = "mhfrt.set_tab"
    bl_label = "Go to Step"
    bl_options = {'INTERNAL'}

    tab: bpy.props.EnumProperty(items=[
        (s, flow.STEP_LABELS[s], "") for s in flow.STEP_IDS])

    @classmethod
    def description(cls, context, properties):
        base = {
            'SETUP': "1 · Setup - load the cage, pick your head mesh",
            'LANDMARKS': "2 · Landmarks - draw matching curves on cage and head",
            'WRAP': "3 · Wrap - fit the cage onto your head",
            'RIG': "4 · Rig - build the skeleton + live control board",
            'TUNE': "5 · Fine-Tune - place eye pivots and tongue bones",
            'BIND': "6 · Bind - transfer weights; the face goes live",
            'PARTS': "7 · Parts - attach eyes, teeth, lashes and your own",
            'MORPHS': "8 · Morphs - pose, pick and sculpt driven morph keys",
            'ANIM': "9 · Animation - import and play a facial performance",
            'EXPORT': "10 · Export - send the character and clips to an engine",
        }[properties.tab]
        req = flow.requirement(context, properties.tab)
        return base + (f".\nLocked: {req[0].lower()}" if req else "")

    def execute(self, context):
        mh = context.scene.mhfrt
        from ..ops import op_pairs
        if op_pairs.is_running():
            self.report({'INFO'},
                        "Finish landmark mode first (Esc in the split view)")
            return {'CANCELLED'}
        if (mh.skeleton is not None
                and mh.skeleton.get(rest_tuning.TONGUE_SESSION_PROP)):
            mh.ui_tab = 'TUNE'
            return {'FINISHED'}
        # No organize pass here any more: the character's collections are
        # created the moment it has a cage or a head, so by the time any step
        # can be opened everything is already filed (see props._obj_update).
        if self.tab == 'MORPHS':
            from ..ops import op_morphs
            # Which mesh the step is on is decided by what is selected NOW, not
            # by a mesh clicked three steps ago: open with a control (or
            # nothing) active and there is no stray mesh to warn about.
            op_morphs.reset_mesh_tracking(mh, context)
            op_morphs.sync_morph_ui_state(mh)
            # The pose may have moved on while another step was open, and the
            # depsgraph prune only runs while this step is the visible one -
            # so the list opens matching the pose, selection and all.
            op_morphs.prune_morph_selection(mh, context)
        mh.ui_tab = self.tab
        return {'FINISHED'}


_HELP = {
    'CHARACTERS': dict(
        icon="logo", title="Rig list",
        sub="Every MHFR rig in this file - each fully independent",
        body=["Every rig you build in this file appears here, and the "
              "highlighted one is the rig the whole panel works on. "
              "Click a name to switch: its cage, head, landmarks, skeleton, "
              "control board, morphs and animation all come back exactly as "
              "you left them - even the step you were on.",
              "Everything is stored per rig. Importing an animation, "
              "sculpting morphs, tuning intensity or attaching parts only "
              "ever touches the selected rig; the others keep playing "
              "their own boards untouched.",
              "+ returns the panel to Setup with empty slots for the next "
              "rig. The Name field renames the selected rig (MHFR by "
              "default). The trash button removes the selected rig - it "
              "deletes everything the add-on made (cage, skeleton, control "
              "board, landmarks, animation, MHFRT collections and props) but "
              "KEEPS your own meshes as plain objects in a new collection.",
              "Pose Control Board jumps straight into Pose Mode on the "
              "selected rig's board; Reset Controllers to Neutral under it "
              "puts every control back - eye-aim handles included - for a "
              "neutral face and a straight-ahead gaze."],
        tips=["A rig appears here the moment its cage and head are both "
              "set - 'setting up' rows are rigs not built yet; switch to "
              "one to continue where you left off.",
              "A film icon on a row means that rig has its own imported "
              "animation clip.",
              "Remove keeps your head and any eyes / teeth / tongue you "
              "made; it only strips the add-on's data off them. It asks "
              "for confirmation first.",
              "Appending a rig's collection into another file brings its "
              "whole rig along, and it shows up in this list there too.",
              "Use Refresh after deleting or appending rigs by hand."],
    ),
    'OVERVIEW': dict(
        icon="logo", title="MetaHuman Rig Transfer",
        sub="Any head mesh -> a live MetaHuman face rig",
        body=["Work through the ten tabs left to right. A tab gets a check "
              "badge when its step is done, and the panel moves forward for "
              "you. You can revisit any step at any time - nothing is ever "
              "destroyed."],
        steps=True,
        tips=["Rig as many characters as you like in one file - the "
              "Characters list at the top switches between them, and "
              "everything (landmarks, morphs, animation) stays per "
              "character.",
              "Every button has a tooltip; every step has this ? dialog.",
              "Everything is undoable with Ctrl+Z, inside tools too.",
              "The display bar under the progress works in every step: "
              "cage / target / both, studio shading (toggle restores your "
              "own shading), cage look, X-ray."],
    ),
    'SETUP': dict(
        icon="tab_setup", title="Step 1 · Setup",
        sub="Two meshes: the cage, and your head",
        body=["Load the bundled Ada cage - pick a LOD first (LOD5 is the "
              "lightest and usually enough; denser cages wrap slower but "
              "hold more detail). Then set your character's head mesh as "
              "the Head Target.",
              "Loading the cage switches the viewport to the monochrome "
              "studio look. The display bar at the top controls it from "
              "any step: show cage / target / both, toggle the studio "
              "shading (off = your previous shading comes back), toggle "
              "the cage's graphite-wireframe look, and X-ray."],
        tips=["Head Cage and Head Target must be two different mesh objects.",
              "Eyes and tongue may be joined into the Head Target; their rig "
              "positions are refined later in Fine-Tune."],
    ),
    'LANDMARKS': dict(
        icon="tab_landmarks", title="Step 2 · Landmarks",
        sub="Draw matching curves on cage and head",
        body=["ZWrap-style split view: the viewport divides in two - the "
              "cage alone on the left, your head alone on the right, each "
              "with its own camera you can orbit and zoom independently.",
              "Hold LMB and draw a curve along a feature on the cage - an "
              "eyebrow, the jaw line, the lip seam. Release, then move the "
              "mouse left or right to choose how many landmarks ride the "
              "curve (just like Ctrl+R - scrolling or typing a number does "
              "the same, so it works on any mouse, trackpad or keymap), "
              "then click or simply start orbiting to confirm. Now draw the "
              "same curve on your head: it lands instantly with the same "
              "number of points, and the matching numbers show the pairing.",
              "Draw both curves in the same direction (the arrow shows it). "
              "After confirming, drag any point to reshape its curve - "
              "points are never added or removed. Esc puts your viewport "
              "back exactly as it was.",
              f"{MIN_PAIRS} points is the minimum - {RECOMMENDED_PAIRS}+ "
              "spread over eyes, nose, mouth, jaw and forehead give the "
              "best wrap."],
        keys=[("LMB drag", "draw a curve"),
              ("Move mouse", "points on the curve"),
              ("Scroll / type a number", "same, if you prefer"),
              ("Click / Orbit", "confirm the cage curve"),
              ("RMB", "cancel the curve"),
              ("Drag point", "reshape a curve"), ("X", "delete hovered curve"),
              ("S", "toggle X symmetry"),
              ("Ctrl+Z", "undo"), ("Esc / RMB", "finish")],
        tips=["Symmetry mirrors the whole curve for you - draw one side "
              "only. Points near the centre line stay single (the small "
              "tick marks them) and are the only place the two sides are "
              "joined into one stroke.",
              "In symmetry, dragging a point moves its mirrored sister too. "
              "Drag the END of a curve onto the centre line and release - "
              "it merges with its sister into one centre landmark and the "
              "two curves join. Only matching curve ends ever merge.",
              "A stroke that crosses the centre line is clipped there - "
              "its last landmark lands exactly on the centre.",
              "If the viewport is too small to split, the tool falls back to "
              "one view that auto-solos the mesh you are drawing on "
              "(H shows both)."],
    ),
    'WRAP': dict(
        icon="tab_wrap", title="Step 3 · Wrap",
        sub="Fit the cage onto your head",
        body=["Warps the cage using your landmarks, then registers it onto "
              "the head surface coarse-to-fine. The result is stored as a "
              "'Wrapped' shape key - the original cage shape stays intact "
              "underneath, forever.",
              "Wrapping again KEEPS your hand refinements: brush and live-"
              "session tweaks are lifted off and re-applied on top of the "
              "new solve. Use Fresh Wrap (in Advanced) to throw them away "
              "and recompute purely from the landmarks.",
              "Refine has two ways to polish the result. LIVE SESSION runs "
              "a softbody simulation that keeps the cage snapped to the "
              "head while you grab and drag the mesh like cloth - drop "
              "pins with Shift+click and the mesh keeps chasing them; all "
              "sliders react in real time. The Slide / Smooth brushes are "
              "the precise, stroke-based alternative."],
        keys=[("Drag", "grab the mesh (live)"), ("Shift+click", "drop a pin (live)"),
              ("X", "delete hovered pin"), ("F  or  [ ]", "brush size"),
              ("Shift+F", "brush strength"),
              ("S / B", "symmetry / borders"), ("Space", "pause simulation"),
              ("Ctrl+Z", "undo"), ("Esc", "finish tool")],
        tips=["Draft quality is great for iterating on landmarks; switch to "
              "High for the final pass.",
              "Region_Mask, Pin Borders and Symmetry all keep working "
              "inside the live session."],
    ),
    'RIG': dict(
        icon="tab_rig", title="Step 4 · Rig",
        sub="Skeleton + live control board",
        body=["Fits the MetaHuman facial skeleton to your wrapped cage and "
              "connects the control board - the rig is live immediately, "
              "driven by RigLogic, and each character keeps its own.",
              "Changed the head, the wrap, or slid vertices afterwards? "
              "Press Update Rig - same skeleton, no re-import, poses "
              "survive. Saved eye and tongue corrections are reapplied.",
              "Choose whether the new face stays Standalone or is Merged Into "
              "the Character. Merging joins the facial bones INTO the "
              "character's own armature, so the character stays ONE skeleton "
              "and exports as one - there is no two-skeleton option.",
              "The merge uses two explicit bones: Parent Bone, which the whole "
              "face follows, and Head Deform Bone, whose painted head weight "
              "the facial bones take their share from. No rig naming "
              "convention is assumed.",
              "Merging also copies the target's body/head/neck weights back "
              "onto the cage, which keeps the cage neck aligned when the "
              "character turns its head. Do it in the neutral pose.",
              "Panel Layout is where the control panels go. The board has "
              "three flat bars - one under the main expressions panel, one "
              "under TWEAKERS, one under the follow-head switches - and each "
              "drags its own panel. Grab Panel Handles selects all three in "
              "Pose Mode, ready for G.",
              "Save Layout remembers the arrangement: it comes back with "
              "Restore, survives Rebuild Control Board, and is written into "
              "the .mhfrt so it follows the character into another file. "
              "Reset Placement starts again beside the head."],
        tips=["Bones Settings, in the Morphs step, scales how strongly the "
              "face moves - for the whole face or one channel at a time.",
              "Select a face control and use I > Location or Blender Auto Key. "
              "Keyed controls evaluate during scrubbing, playback, and render.",
              "The fitted armature is colour-coded by facial region. Open "
              "Armature Data > Bone Collections to isolate Eyes, Tongue, "
              "Mouth, and the other regions. Dense 12IPV correctives start "
              "hidden under each region.",
              "The panel bars drive nothing - dragging one moves its panel "
              "and never the face - so Reset Controllers to Neutral, clearing "
              "an animation clip and exporting all leave them where you put "
              "them."],
    ),
    'ANIM': dict(
        icon="tab_rig", title="Step 9 · Animation",
        sub="Play a UE5 performance on this character",
        body=["Import a facial performance exported from Unreal Engine - "
              "the clip lands as keyframes on this character's own control "
              "board, so the face plays through the live rig and you can "
              "still grab any control and polish on top.",
              "FBX - in Unreal, bake the performance to the face control "
              "rig in a Level Sequence, then right-click the "
              "Face_ControlBoard_CtrlRig track > Export. Skeletal-mesh "
              "animation exports (bone tracks only) are not facial curves.",
              "CSV - a raw take recorded with the Live Link Face iPhone "
              "app (ARKit). The 52 blend shapes are mapped onto the "
              "matching RigLogic controls.",
              "JSON - a plain curve dump: {\"fps\": 30, \"curves\": "
              "{name: [[frame, value], ...]}} with board control, "
              "CTRL_expressions or ARKit names.",
              "Export Face Animation writes the current board motion to a "
              ".mhfa clip - the add-on's own format (readable JSON inside, "
              "re-imported here). Use it to back up a take, move it to "
              "another character, or hand-edit the curves."],
        tips=["Importing replaces the previous clip; Remove resets the "
              "board to rest. Both are undoable.",
              "Export captures your hand-keyed polish too, not just an "
              "imported clip - anything keyed on the board is saved.",
              "For matching body and face clips, use names such as Walk and "
              "FaceAnim_Walk.",
              "The scene frame range is set to the clip length; timing is "
              "kept exact even when frame rates differ.",
              "Timeline animation on controls also drives sculpted morph "
              "keys during playback and render."],
    ),
    'EXPORT': dict(
        icon="tab_rig", title="Step 10 · Export",
        sub="Engine-ready character + corrective animation",
        body=["Unity export writes one FBX containing the final armature, "
              "driven character meshes and every animation clip. Each clip "
              "contains visually baked body/facial bones and the matching "
              "corrective blend-shape curves.",
              "Unreal export follows Epic's one-animation-per-FBX workflow. "
              "It writes one Skeletal Mesh FBX with every morph target, then "
              "an Animations folder with one FBX per clip and its matching "
              "corrective morph-target curves.",
              "Both exporters work from the real armature hierarchy and do "
              "not assume Rigify. The cage, face board and add-on helpers are "
              "excluded. Temporary bakes are removed and the Blender file is "
              "restored after export.",
              "Bake to Drivers writes a finished .blend that needs no add-on. "
              "The control board comes along and keeps working, because the "
              "DNA's control -> raw -> corrective network is rebuilt as real "
              "Blender drivers on the skeleton. The joint half cannot be "
              "drivers - its tables hold nearly a million coefficients - so "
              "the bone motion becomes one shape key per RigLogic input, "
              "combination correctives included, each driven by that network. "
              "The meshes keep their weights and Armature modifier, so body "
              "and neck animation still work.",
              "The CTRL_C_eyesAim look-at is not a DNA channel - it is solved "
              "in Python - so it is rebuilt out of Damped Track constraints on "
              "helper bones, and its answer re-enters the eye channels the way "
              "the live rig writes it. The eyelids and the eye-look correctives "
              "therefore follow the aim handles too, and CTRL_lookAtSwitch "
              "still blends and keyframes.",
              "The centre-eye master and every control's Limit Location frame "
              "are carried into the driver expressions, so the delivered board "
              "behaves like the one you animated on."],
        tips=["For paired body and face animation, use names such as Walk and "
              "FaceAnim_Walk.",
              "In Unreal, import the Skeletal Mesh FBX first with Morph "
              "Targets enabled. Then import the animation FBXs using that "
              "Skeleton with Import Animations enabled.",
              "An Existing Armature character must be merged into one "
              "armature before either game-engine export.",
              "A baked driver rig is a blend-shape face: eye look travels a "
              "chord rather than an arc. Turning Bake Eye Look off drops the "
              "look-at with it, since both drive the same channels.",
              "Bake to Drivers with Controls Only for a much smaller file, at "
              "the cost of the combination correctives."],
    ),
    'TUNE': dict(
        icon="tab_rig", title="Step 5 · Fine-Tune",
        sub="Exact eye pivots + target tongue fit",
        body=["This step changes rest-bone positions only. It never moves the "
              "target geometry and it does not create weights.",
              "Each eye bone is a ROTATION PIVOT: the rig spins the eye "
              "around it, so it belongs at the center of the eyeball sphere - "
              "the point the eye actually turns about. Where the eye mesh's "
              "Object Origin sits does not matter to the rig; it is wherever "
              "the mesh happened to be modelled from, and Origin to Geometry "
              "only moves it to the median of the vertices, which the cornea "
              "bulge drags forward. Select the eyeball vertices, Shift+S > "
              "Cursor to Selected, check the cursor in front and side views, "
              "then place that eye.",
              "For the tongue, the add-on isolates every purple tongue bone "
              "and hands them to you already selected - all of them, so S, R "
              "and G move the tongue as one piece with no select-all first. "
              "Shape them around the target tongue, then Finish. The "
              "correction survives Update Rig and RigLogic motion follows the "
              "adjusted direction and scale."],
        keys=[("Shift+S", "Cursor to selected eye vertices"),
              ("S", "uniformly scale tongue"),
              ("R", "rotate tongue"),
              ("G", "move tongue")],
        tips=["Character Left is the character's own left (viewer's right).",
              "On a full eyeball sphere, Cursor to Selected over all its "
              "vertices lands on the center. On a half or heavily bulged eye "
              "it lands slightly forward - nudge the cursor back until it "
              "looks centered in both front and side views.",
              "Use Alt+Z X-Ray or hide head vertices if the tongue is inside "
              "a closed mouth.",
              "Use only G, R, and S on tongue bones. If a bone is deleted, "
              "renamed, or reparented, press Ctrl+Z before Finish or Cancel."],
    ),
    'BIND': dict(
        icon="tab_bind", title="Step 6 · Bind",
        sub="Weights over, character live",
        body=["Copies the skin weights from the wrapped cage onto your head "
              "and binds it to the skeleton. After this, grab any control "
              "on the board and the face follows.",
              "EYE AIM: the target floating in front of the face steers the "
              "eyes. CTRL_lookAtSwitch on the board is a BLEND, not an on/off: "
              "at 0 the eye controls behave normally and the target does "
              "nothing, at 1 the eyes point straight at it, and part way up "
              "they sit between the two - so you can ease into a look, or "
              "keyframe the handle to hand the eyes over and take them back. "
              "Move CTRL_C_eyesAim to aim both eyes, or CTRL_L_eyeAim / "
              "CTRL_R_eyeAim for one at a time; the eyes hold their target "
              "when the head turns.",
              "Only facial groups are replaced. Existing body vertex "
              "groups and its Armature modifier are left untouched.",
              "Weight Cleanup (optional) fixes weight bleed where lips or "
              "eyelids touch: open the mouth / close the eyes with one "
              "slider - your head follows automatically - then Re-Bind "
              "in that pose."],
        tips=["You can re-bind as often as you like - weights are simply "
              "recomputed.",
              "Joined eyes and tongue can be weight-painted after their bones "
              "have been positioned in Fine-Tune."],
    ),
    'CLEANUP': dict(
        icon="polish", title="Weight Cleanup",
        sub="Optional - for sealed lips / closed eyes",
        body=["Where lips or eyelids touch, weights can bleed across (top "
              "lip follows the jaw). The fix: separate the surfaces, then "
              "bind in that pose.",
              "Set Mouth Open or Eyes Closed and press Re-Bind - that is the "
              "whole workflow. The sliders are free to drag: nothing is "
              "computed and no mesh moves until the bind, which poses the "
              "cage AND your head behind the scenes, transfers the weights "
              "across the separated surfaces and returns both to neutral.",
              "Your head's pose is generated from the wrapped cage, scaled to "
              "your head's size, using the same surface matching the weight "
              "transfer uses. Nothing to sculpt, no masks, no Edit mode.",
              "Pose Meshes To Sliders is optional - press it only when you "
              "want to SEE the pose, to sculpt or smooth-brush the lips in "
              "it. Re-Bind then uses your sculpted version. It behaves "
              "identically whether or not you ever press it."],
        tips=["Refined the wrap afterwards? Nothing to redo - the head's pose "
              "keys regenerate themselves at the next bind.",
              "The landmarks ride the posed cage, so you can sanity-check "
              "the pose at a glance."],
    ),
    'PARTS': dict(
        icon="tab_bind", title="Step 7 · Parts",
        sub="Eyes, teeth, eyelashes, tongue and your own parts",
        body=["Binds the separate accessory meshes you made to the facial "
              "rig in one click each. Select your mesh in the viewport, "
              "then press its slot.",
              "Each attach is a clean re-bind: old armature modifiers are "
              "removed, the mesh is parented to the facial skeleton, and "
              "its vertex groups are rebuilt. Eyes and teeth follow one "
              "bone at 100%. Eyelashes sample the Head Target at their "
              "head-facing roots, then spread those weights through their "
              "own connected topology without tip falloff. The tongue keeps "
              "automatic weights from its tongue bones only.",
              "Need a slot we did not name? Type your own under Custom - "
              "brows, peach fuzz, stubble cards, a piercing - and press Add. "
              "A custom part binds exactly like an eyelash: its head-facing "
              "roots sample the Head Target's skin weights, which then spread "
              "through the mesh, so it follows whatever deforms the face "
              "underneath it. Your named slots stay listed and re-attach in "
              "one click.",
              "A posed character is no problem: attaching switches both "
              "rigs to Rest Position, binds, and restores the pose. Place "
              "the mesh where it belongs on the REST-pose head - it snaps "
              "into the pose the moment it is attached.",
              "This step is optional: skip it if eyes, teeth, and tongue "
              "are joined into the Head Target."],
        tips=["Use the Rest Position toggle in Armature Data to see the "
              "rest-pose head while placing a part on a posed character.",
              "You can re-attach any time - the previous binding is "
              "replaced cleanly.",
              "Several selected meshes attach together to the same slot "
              "(e.g. upper teeth + gums).",
              "Character Left is the character's own left (viewer's "
              "right)."],
    ),
    'MORPHS': dict(
        icon="polish", title="Step 8 · Morphs",
        sub="Pose, pick, sculpt - keys manage themselves",
        body=["Pose the face controls: the list shows the DNA morph "
              "channels that pose activates, strongest first. Click one to "
              "select it (highlighted blue) - its shape key is created "
              "automatically, and keys you never sculpt remove themselves. "
              "The mesh only ever carries the shapes you actually sculpted.",
              "The list is a live view of the pose, nothing more. Pose "
              "elsewhere and the rows follow the new pose; a neutral face "
              "activates nothing, so the list is simply EMPTY with nothing "
              "selected. That is its resting state, not a problem - no row is "
              "ever kept selected once the pose stops firing it. Only "
              "channels you dialled down or switched off by hand keep their "
              "rows, so you can always find them again.",
              "RigLogic drives the key values live while you work. The "
              "buttons beside the list delete driven keys - minus for the "
              "selected row, then Compact Untouched (empty keys only) and "
              "Remove All Morphs; pick a morph again to recreate it empty.",
              "The BONE ICON on a row previews that morph WITHOUT its bones, "
              "so you can judge the sculpted shape on its own. It is only a "
              "viewing state: the channel's value and its Channels Intensity "
              "are untouched, and the row keeps its place in the list. The "
              "'No Bones' toggle above the list narrows the list to everything "
              "you have switched off that way, and Bring Bones Back restores "
              "them all at once.",
              "BONES SETTINGS sets how strongly the rig moves BONES. "
              "Expressions Intensity scales the whole face; Channels "
              "Intensity scales only the morph row you selected. Either one "
              "at 0 stops the bone movement while the shape keys keep "
              "deforming - so a channel can run on your sculpted shape "
              "alone. The board controllers never move, and both settings "
              "are stored per character. Channels Intensity only appears "
              "while a row is selected, because it dials that one channel.",
              "OBJECTS decides WHICH mesh you sculpt. Each row says what the "
              "mesh IS before its name - Head, the cage, or the slot it was "
              "attached to in the Parts step (Tongue, Lash Upper Right, and so "
              "on) - because a list of shirt.015 / shirt.016 tells you nothing "
              "about which one you are opening. The list and the "
              "viewport always agree: click a row and that mesh is selected in "
              "the 3D view; click the mesh in the 3D view and its row lights "
              "up. Only meshes count - clicking a control board bone or a "
              "landmark curve leaves your choice alone, so you can pose and "
              "then sculpt. Click a mesh that is NOT in the list and the "
              "morphs disappear behind a warning naming it: add it with the "
              "button there, or pick a mesh from the list again. Nothing in "
              "this step ever sculpts a mesh you did not put in the list.",
              "Reset Controllers to Neutral, under Pose Control Board in the "
              "Rig list, puts every board control - the eye-aim handles "
              "included - back to its rest pose, which empties the list and "
              "clears the selection with it."],
        tips=["The strongest active morphs appear first.",
              "Use the sculpt / edit icons on a row to jump straight into "
              "that mode.",
              "Turn off 'Custom' to focus on the standard DNA morphs; the "
              "bone-mute icon only appears on morphs that move bones.",
              "The bone-mute icon on a row is the same setting as Channels "
              "Intensity at 0 - a shortcut for muting several channels "
              "quickly. Restore All Channels puts them all back to full.",
              "A muted channel still previews while you sculpt it, so muting "
              "bones first is a clean way to sculpt a corrective shape."],
    ),
}


class MHFRT_OT_help(bpy.types.Operator):
    """Explain this step: what it does, how to use it, hotkeys and tips."""
    bl_idname = "mhfrt.help"
    bl_label = "Help"
    bl_options = {'INTERNAL'}

    topic: bpy.props.EnumProperty(items=[
        (k, k.title(), "") for k in _HELP])

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=340)

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        t = _HELP[self.topic]
        layout = self.layout

        row = layout.row()
        row.template_icon(icon_value=icons.icon(t["icon"]), scale=2.8)
        col = row.column(align=True)
        col.label(text=t["title"])
        sub = col.row()
        sub.enabled = False
        sub.label(text=t["sub"])
        layout.separator()

        for para in t["body"]:
            col = layout.column(align=True)
            for line in textwrap.wrap(para, _WRAP_CHARS):
                col.label(text=line)
            layout.separator(factor=0.6)

        if t.get("steps"):  # overview: the workflow tabs, one line each
            box = layout.box()
            col = box.column(align=True)
            for i, s in enumerate(flow.STEP_IDS):
                col.label(text=f"{i + 1}   {flow.STEP_LABELS[s]}",
                          **_tab_icon_kwargs(s))
            layout.separator(factor=0.6)

        if t.get("keys"):
            box = layout.box()
            col = box.column(align=True)
            for key, action in t["keys"]:
                split = col.split(factor=0.38)
                split.label(text=key)
                split.label(text=action)
            layout.separator(factor=0.6)

        for tip in t.get("tips", ()):
            col = layout.column(align=True)
            for i, line in enumerate(textwrap.wrap(tip, _WRAP_CHARS - 3)):
                col.label(text=line, icon='INFO' if i == 0 else 'BLANK1')


# ------------------------------------------------------------ draw helpers ---

def _section(layout, mh, prop_name, title, icon='NONE', icon_value=0):
    """Dark disclosure box: triangle + title header; body column when open."""
    box = layout.box()
    row = box.row(align=True)
    is_open = getattr(mh, prop_name)
    row.prop(mh, prop_name, text="",
             icon='TRIA_DOWN' if is_open else 'TRIA_RIGHT', emboss=False)
    if icon_value:
        row.prop(mh, prop_name, text=title, icon_value=icon_value, emboss=False)
    else:
        row.prop(mh, prop_name, text=title, icon=icon, emboss=False)
    return box.column(align=False) if is_open else None


def _locked(card, req):
    """A locked step shows one line + a jump button - nothing else."""
    msg, fix_tab = req
    col = card.column(align=True)
    row = col.row()
    row.enabled = False
    row.label(text=msg, icon='LOCKED')
    op = col.operator("mhfrt.set_tab", icon='LOOP_BACK',
                      text=f"Go to {flow.STEP_LABELS[fix_tab]}")
    op.tab = fix_tab


def _help_btn(row, topic):
    sub = row.row(align=True)
    sub.alignment = 'RIGHT'
    sub.emboss = 'NONE'
    sub.operator("mhfrt.help", text="", icon='QUESTION').topic = topic


def _tab_icon_kwargs(step_id, done=False):
    icon_value = icons.tab_icon(step_id, done)
    if icon_value:
        return {"icon_value": icon_value}
    if step_id == 'TUNE':
        return {"icon": 'CHECKMARK' if done else 'PIVOT_CURSOR'}
    if step_id == 'PARTS':
        return {"icon": 'CHECKMARK' if done else 'GROUP_BONE'}
    if step_id == 'MORPHS':
        return {"icon": 'SHAPEKEY_DATA' if not done else 'CHECKMARK'}
    if step_id == 'ANIM':
        return {"icon": 'RENDER_ANIMATION' if not done else 'CHECKMARK'}
    if step_id == 'EXPORT':
        return {"icon": 'EXPORT' if not done else 'CHECKMARK'}
    return {"icon": 'BLANK1'}


# ------------------------------------------------------------- characters ---

class MHFRT_UL_rigs(bpy.types.UIList):
    """Every MHFR rig in the file, as a Morphs-style selectable list.

    Each row is an operator button (like the Morphs Objects list) so clicking
    it makes that rig the one the whole panel works on; the template_list's own
    highlight marks the active row.  A film badge flags a rig that carries an
    imported animation clip."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        if item.is_new:
            # an in-Setup pair with no rig yet: jump back to Setup to finish it
            op = row.operator("mhfrt.set_tab",
                              text=f"{item.rig_name}  ·  setting up",
                              icon='USER', emboss=False)
            op.tab = 'SETUP'
            return
        label = item.rig_name
        icon_id = 'ARMATURE_DATA'
        if item.is_setup:
            label = f"{item.rig_name}  ·  setting up"
            icon_id = 'USER'
        op = row.operator("mhfrt.activate_character", text=label,
                          icon=icon_id, emboss=False)
        op.skeleton_name = item.skel_name
        op.root_name = item.root_name
        op.rig_key = item.rig_key
        op.list_index = index
        if item.has_anim:
            row.label(text="", icon='RENDER_ANIMATION')


def _wrap_note(text, width=46):
    """Break one repair message into panel-width lines.

    Object names arrive in these messages and can be longer than the panel on
    their own, so an oversized word is cut rather than allowed to stretch the
    region.
    """
    words = []
    for word in str(text).split():
        while len(word) > width:
            words.append(word[:width])
            word = word[width:]
        if word:
            words.append(word)
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _draw_characters(layout, context, mh):
    """The Rig list: every MHFR rig in the file as a Morphs-style template_list.

    Hidden while the file has no rig and nothing is being set up. Selecting a
    row makes that rig the one the whole panel operates on; the side column
    adds a new rig, removes the selected one, or re-scans the file."""
    from ..ops import op_rig

    # Refresh the backing collection outside draw (deferred timer) - never
    # mutate it here.
    op_rig.request_rig_ui_sync(mh, context)
    # Follow the artist's selection into this list.  It belongs here rather
    # than in the depsgraph handler alone because selecting an object produces
    # no depsgraph update - but it does redraw this panel.  Draw-safe: the
    # switch itself runs from a timer.
    op_rig.queue_character_pick(context, mh)

    # The box is ALWAYS drawn, even with nothing in it.  Hiding it on an empty
    # file also hid the only way to LOAD a saved character - you could not open
    # a .mhfrt until you had first made a rig by hand, which is backwards.
    box = layout.box()
    head = box.row(align=True)
    head.label(text=f"Rig list  ·  {len(mh.rig_ui_items)}",
               icon='OUTLINER_OB_ARMATURE')
    sub = head.row(align=True)
    sub.alignment = 'RIGHT'
    sub.emboss = 'NONE'
    sub.operator("mhfrt.help", text="", icon='QUESTION').topic = 'CHARACTERS'

    row = box.row()
    row.template_list(
        "MHFRT_UL_rigs", "",
        mh, "rig_ui_items",
        mh, "rig_ui_active_index",
        rows=3,
    )
    col = row.column(align=True)
    col.operator("mhfrt.new_character", text="", icon='ADD')
    remove = col.column(align=True)
    remove.enabled = 0 <= mh.rig_ui_active_index < len(mh.rig_ui_items)
    remove.operator("mhfrt.remove_rig", text="", icon='TRASH')

    idx = mh.rig_ui_active_index
    has_row = 0 <= idx < len(mh.rig_ui_items)
    if has_row:
        box.prop(mh, "active_rig_name", text="Name")

    # A live rig is posed from Python: during a render Blender writes its bones
    # from the render thread, and only Lock Interface keeps the main thread off
    # the same depsgraph while it does (see core.render_state). The add-on
    # switches it on by itself unless preferences say not to - so if it is off
    # here, the artist chose that, and should know what it costs.
    from ..core import render_state
    if op_rig.has_live_rigs() and render_state.lock_needed(context.scene):
        warn = box.box()
        warn.alert = True
        warn.label(text="Lock Interface is off", icon='ERROR')
        col = warn.column(align=True)
        col.scale_y = 0.8
        col.enabled = False
        for line in _wrap_note("Rendering a live rig without it can take "
                               "Blender down partway through the frames."):
            col.label(text=line)
        warn.prop(context.scene.render, "use_lock_interface")

    # Anything the add-on repaired on its own says so here. A silent repair is
    # how "the rig stopped controlling the character" comes back a second time
    # with nobody able to say what changed.
    notes = op_rig.repair_notes()
    if notes:
        card = box.box()
        head = card.row(align=True)
        head.label(text="Duplicate sorted out", icon='INFO')
        sub = head.row(align=True)
        sub.alignment = 'RIGHT'
        sub.emboss = 'NONE'
        sub.operator("mhfrt.dismiss_repair_notes", text="", icon='X')
        col = card.column(align=True)
        col.scale_y = 0.8
        col.enabled = False
        for text in notes:
            for line in _wrap_note(text):
                col.label(text=line)

    # A character restored from a .mhfrt keeps its landmark curves parked until
    # its head mesh is pointed at.  Say so, plainly, instead of leaving the
    # artist wondering where their work went.
    from ..core import registry
    record = registry.active(context.scene)
    if record is not None and record.pending_landmarks:
        note = box.box()
        note.alert = True
        wanted = record.pending_target_name or "its head mesh"
        note.label(text="Waiting for this character's head", icon='ERROR')
        col = note.column(align=True)
        col.scale_y = 0.8
        col.label(text=f"Saved as '{wanted}' - not in this file.")
        col.label(text="Pick it as Head Target in Setup;")
        col.label(text="the landmark curves come back with it.")

    # A rig is hours of work.  It should not live only inside one .blend.
    save = box.row(align=True)
    sub = save.row(align=True)
    sub.enabled = has_row
    sub.operator("mhfrt.save_character", text="Save .mhfrt", icon='FILE_TICK')
    # Never disabled: loading is how you get your FIRST character back.
    save.operator("mhfrt.load_character", text="Load .mhfrt",
                  icon='FILEBROWSER')

    # Where this character's file IS.  It was written the moment the row was
    # created and it holds the record of everything the add-on added to the
    # scene - including the artist's own armature as it was before a merge -
    # so it is worth being able to see, and worth shouting about if the write
    # failed.
    if record is not None:
        if record.file_error:
            warn = box.box()
            warn.alert = True
            warn.label(text="Character file could not be written",
                       icon='ERROR')
            col = warn.column(align=True)
            col.scale_y = 0.8
            for line in _wrap_note(record.file_error):
                col.label(text=line)
            col.label(text="Set a writable Character File Folder in the")
            col.label(text="add-on preferences - Remove needs this file")
        elif record.file_path:
            info = box.row(align=True)
            info.enabled = False
            info.scale_y = 0.8
            info.label(text=os.path.basename(record.file_path),
                       icon='CHECKMARK')

    # Straight to the selected rig's board in Pose Mode. Greys out on a rig
    # with no bone board (pre-3.0 loose objects) - see the operator's poll.
    col = box.column(align=True)
    col.operator("mhfrt.edit_board", icon='POSE_HLT')
    # Posing is what puts the board out of neutral, so the way back belongs
    # next to the way in - not buried in the step that happens to use it.
    col.operator("mhfrt.reset_all_controllers", icon='LOOP_BACK',
                 text="Reset Controllers to Neutral")


# ------------------------------------------------------------ tab content ---

def _draw_setup(card, context, mh):
    if (mh.skeleton is not None
            and mh.skeleton.get(rest_tuning.TONGUE_SESSION_PROP)):
        _locked(card, ("Finish or Cancel Tongue Edit", 'TUNE'))
        return
    col = card.column(align=True)
    col.prop(mh, "cage_lod", text="")
    sub = col.column(align=True)
    sub.scale_y = 1.3
    sub.operator("mhfrt.load_cage", icon='IMPORT')

    # Native Object fields - the exact same widget as the Armature modifier's
    # "Object" input: a browse dropdown, a viewport EYEDROPPER (pick a mesh by
    # clicking it in the 3D view), the name, and a clear X.  The property poll
    # (`_is_mesh`) limits both the list and the eyedropper to mesh objects, so
    # the artist can only point the wrap at real geometry.
    col = card.column(align=True)
    col.prop(mh, "cage", text="Head Cage")
    col.prop(mh, "target", text="Head Target")


def _draw_landmarks(card, context, mh):
    req = flow.requirement(context, 'LANDMARKS')
    if req:
        _locked(card, req)
        return
    n = lmdata.complete_count(mh)
    ncurves = lmdata.curve_count(mh)

    col = card.column()
    col.scale_y = 1.4
    col.operator("mhfrt.edit_landmarks", icon='GREASEPENCIL',
                 text="Draw Landmark Curves")

    row = card.row(align=True)
    row.prop(mh, "symmetry", toggle=True, icon='MOD_MIRROR')
    row.prop(mh, "landmark_sync_view", toggle=True, icon='LINKED')
    row = card.row(align=True)
    row.prop(mh, "landmark_lazy", toggle=True, icon='BRUSH_DATA')
    sub = row.row(align=True)
    sub.enabled = mh.landmark_lazy
    sub.prop(mh, "landmark_lazy_radius", text="")

    status = f"{ncurves} curves  ·  {n} points"
    if 0 < n < RECOMMENDED_PAIRS:
        status += f"  ·  {RECOMMENDED_PAIRS}+ recommended"
    row = card.row(align=True)
    row.label(text=status, icon='GROUP_VERTEX')
    if n:
        row = card.row(align=True)
        row.operator("mhfrt.clear_pairs", text="Clear All", icon='TRASH')

    body = _section(card, mh, "ui_adv_landmarks", "Advanced", icon='TOOL_SETTINGS')
    if body:
        body.prop(mh, "symmetry_center_threshold", slider=True)
        body.prop(mh, "landmark_loop_merge_threshold", slider=True)
        sub = body.column(align=True)
        sub.enabled = mh.landmark_lazy
        sub.prop(mh, "landmark_lazy_radius", slider=True)
        sub.prop(mh, "landmark_lazy_smooth", slider=True)


def _draw_wrap(card, context, mh):
    req = flow.requirement(context, 'WRAP')
    if req:
        _locked(card, req)
        return
    wrapped = flow.wrap_done(mh)
    cage = mh.cage

    row = card.row(align=True)
    row.prop(mh, "wrap_quality", expand=True)
    col = card.column()
    col.scale_y = 1.4
    col.operator("mhfrt.wrap", icon='MOD_SHRINKWRAP', text="Wrap Head Cage to Head")

    if wrapped:
        from ..ops import op_live
        live = op_live.is_running()
        body = _section(card, mh, "ui_sec_refine", "Refine", icon='BRUSH_DATA')
        if body:
            row = body.row(align=True)
            row.scale_y = 1.2
            if live:
                row.operator("mhfrt.live_session", icon='CANCEL',
                             text="Stop Session", depress=True)
                row.prop(mh, "live_pause", text="",
                         icon='PLAY' if mh.live_pause else 'PAUSE')
            else:
                row.operator("mhfrt.live_session", icon='PLAY',
                             text="Live Session")
            if live:
                col = body.column(align=True)
                col.prop(mh, "live_stiffness", slider=True)
                col.prop(mh, "live_smooth", slider=True)
                col.prop(mh, "live_snap", slider=True)
                col.prop(mh, "live_untangle", slider=True)
                col.prop(mh, "live_pin_force", slider=True)
                col.prop(mh, "live_keep_outside", toggle=True)
                sub = body.row()
                sub.enabled = False
                sub.label(text="Drag = grab  ·  Shift+click = pin  ·  Esc done")
            row = body.row(align=True)
            row.scale_y = 1.2
            op = row.operator("mhfrt.slide_brush", icon='BRUSH_DATA', text="Slide")
            op.mode = 'SLIDE'
            op = row.operator("mhfrt.slide_brush", icon='MOD_SMOOTH', text="Smooth")
            op.mode = 'SMOOTH'
            row.prop(mh, "brush_pin_boundary", text="", icon='PINNED')
            col = body.column(align=True)
            col.prop(mh, "brush_radius", text="Size", slider=True)
            col.prop(mh, "brush_strength", text="Strength", slider=True)
            if not live:
                from ..ops.op_wrap import WRAPPED_KEY
                row = body.row(align=True)
                row.prop(cage.data.shape_keys.key_blocks[WRAPPED_KEY], "value",
                         text="Wrapped", slider=True)
                row.operator("mhfrt.wrap_reset", text="", icon='LOOP_BACK')

    body = _section(card, mh, "ui_adv_wrap", "Advanced", icon='TOOL_SETTINGS')
    if body:
        col = body.column(align=True)
        col.prop(mh, "wrap_use_icp")
        sub = col.column(align=True)
        sub.enabled = mh.wrap_use_icp
        sub.prop(mh, "wrap_use_region_mask")
        sub.prop(mh, "wrap_pin_landmarks")
        sub.prop(mh, "wrap_maxdist_frac")
        if mh.wrap_quality == 'CUSTOM':
            sub.separator()
            sub.prop(mh, "wrap_iterations")
            sub.prop(mh, "wrap_step")
            sub.prop(mh, "wrap_smooth")
        col.separator()
        col.prop(mh, "brush_strength")
        if flow.wrap_done(mh):
            col.separator()
            op = col.operator("mhfrt.wrap", icon='FILE_REFRESH',
                              text="Fresh Wrap - Discard Refinements")
            op.fresh = True


def _deform_bone_ok(body, deform, target, parent_bone=""):
    """Is this a usable Head Deform Bone?

    Cheap enough for a redraw: group NAMES only, never weights. A bone named
    "head" that is an empty hierarchy node counts when the bones under it are
    painted - the merge takes the face's share from that whole region, so that
    IS the head bone (see op_merge.head_region_bones).
    """
    if deform is None or not deform.use_deform or target is None:
        return False
    if target.vertex_groups.get(deform.name) is not None:
        return True
    from ..ops.op_merge import head_region_bones
    region = head_region_bones(body, parent_bone or deform.name,
                               deform.name, target)
    return bool(region - {deform.name})


def _draw_rig(card, context, mh):
    if (mh.skeleton is not None
            and mh.skeleton.get(rest_tuning.TONGUE_SESSION_PROP)):
        _locked(card, ("Finish or Cancel Tongue Edit", 'TUNE'))
        return
    req = flow.requirement(context, 'RIG')
    if req:
        _locked(card, req)
        return
    skel = mh.skeleton
    has_rig = flow.rig_built(mh)

    col = card.column()
    col.scale_y = 1.4
    col.operator("mhfrt.fit_skeleton", icon='ARMATURE_DATA',
                 text="Update Rig" if has_rig else "Build Rig")

    if has_rig:
        row = card.row(align=True)
        row.label(text=f"{skel.name}  ·  live", icon='CHECKMARK')

        # Rigs built before 3.0 still carry the loose-object board. Offer the
        # bone board rather than silently leaving them on the old one.
        from ..ops import op_rig as _op_rig
        if _op_rig.board_is_legacy(skel):
            legacy = card.box()
            legacy.label(text="This character is on the old object board",
                         icon='INFO')
            legacy.operator("mhfrt.rebuild_board", icon='BONE_DATA')

        _draw_board_layout(card, mh, skel)

    elif skel:
        card.label(text=f"Skeleton: {skel.name}", icon='ARMATURE_DATA')

    if has_rig:
        from ..props import (RIG_MERGED_PROP, RIG_MERGE_BONE_PROP,
                             RIG_DESTINATION_PROP)

        box = card.box()
        box.label(text="How should this rig follow the character?",
                  icon='CONSTRAINT_BONE')
        col = box.column(align=True)

        if skel.get(RIG_MERGED_PROP):
            bone = str(skel.get(RIG_MERGE_BONE_PROP, ""))
            col.label(text=f"One armature: {skel.name}", icon='CHECKMARK')
            hint = col.column(align=True)
            hint.enabled = False
            if bone:
                hint.label(text=f"Parent Bone: {bone}")
            hint.label(text="Exports as one skeleton")
            deform_bone = skel.data.bones.get(mh.merge_deform_bone)
            deform_ok = (not (deform_bone is not None
                              and deform_bone.name.startswith("FACIAL_"))
                         and _deform_bone_ok(skel, deform_bone, mh.target,
                                             bone))
            select = col.row(align=True)
            select.alert = not deform_ok
            select.prop_search(
                mh, "merge_deform_bone", skel.data, "bones",
                text="Head Deform Bone")
            row = col.row(align=True)
            row.scale_y = 1.2
            row.enabled = deform_ok
            row.operator("mhfrt.merge_body_rig", icon='FILE_REFRESH',
                         text="Refresh Merged Rig")
            return

        col.prop(mh, "rig_destination", expand=True)
        confirmed = str(skel.get(RIG_DESTINATION_PROP, ""))
        if mh.rig_destination == 'UNDECIDED':
            hint = col.row()
            hint.alert = True
            hint.label(text="Choose one option before continuing", icon='INFO')
            return

        if mh.rig_destination == 'STANDALONE':
            row = col.row()
            row.scale_y = 1.25
            row.operator(
                "mhfrt.confirm_standalone_rig",
                icon='CHECKMARK' if confirmed == 'STANDALONE' else 'ARMATURE_DATA',
                text=("Standalone Confirmed"
                      if confirmed == 'STANDALONE'
                      else "Use Standalone Facial Rig"),
            )
            hint = col.row()
            hint.enabled = False
            hint.label(text="No character armature or body-weight copy")
            return

        # Not standalone means ONE skeleton: the facial bones are joined into
        # the character's own armature. There is no two-skeleton option.
        col.separator()
        col.label(text="Merge Into Character Armature",
                  icon='OUTLINER_OB_ARMATURE')
        col.prop(mh, "merge_body_armature", text="Character Armature")
        body = mh.merge_body_armature
        parent_ok = False
        deform_ok = False
        if body and body.type == 'ARMATURE':
            parent_ok = bool(mh.merge_head_bone
                             and mh.merge_head_bone in body.data.bones)
            sub = col.row(align=True)
            sub.alert = not parent_ok
            sub.prop_search(mh, "merge_head_bone", body.data, "bones",
                            text="Parent Bone")
            deform = body.data.bones.get(mh.merge_deform_bone)
            deform_ok = _deform_bone_ok(body, deform, mh.target,
                                        mh.merge_head_bone)
            sub = col.row(align=True)
            sub.alert = not deform_ok
            sub.prop_search(mh, "merge_deform_bone", body.data, "bones",
                            text="Head Deform Bone")
        else:
            col.label(text="Pick the character's own armature", icon='INFO')

        row = col.row(align=True)
        row.scale_y = 1.3
        row.enabled = parent_ok and deform_ok
        row.operator("mhfrt.merge_body_rig", icon='GROUP_BONE',
                     text="Merge Into One Armature")
        hint = col.column(align=True)
        hint.enabled = False
        if parent_ok and deform_ok:
            hint.label(text=f"Facial bones go under {mh.merge_head_bone}")
            hint.label(text=f"Facial share comes from {mh.merge_deform_bone}")
            hint.label(text="Use the neutral pose", icon='INFO')
        else:
            hint.label(text="Choose both bones; they may be different")


def _draw_board_layout(card, mh, skel):
    """Where this character's three control panels sit, and how to keep it.

    Lives in the Rig step because the board is what this step builds, and it is
    closed by default: an artist who never moves a panel never has to see it.
    """
    from ..core import board as boardmod
    from ..ops import op_rig as _op_rig

    body = _section(card, mh, "ui_sec_board", "Panel Layout",
                    icon='SNAP_FACE_CENTER')
    if body is None:
        return

    arm_obj = _op_rig.board_armature(skel)
    handles = boardmod.layout_handle_bones(arm_obj)
    if not handles:
        row = body.row()
        row.enabled = False
        row.label(text="This board has no panel bars", icon='INFO')
        return

    placing = boardmod.layout_unlocked(arm_obj)

    col = body.column(align=True)
    col.scale_y = 1.2
    col.operator("mhfrt.board_layout", icon='VIEW_PAN',
                 text="Grab Panel Handles").action = 'SELECT'

    # The board is locked except while placing, so which state it is in decides
    # whether G/R/S do anything at all. Say it rather than let the artist
    # discover it by pressing G and watching nothing happen.
    if placing:
        mode = body.box()
        mode.alert = True
        mode.label(text="Redesigning - board unlocked", icon='UNLOCKED')
        sub = mode.column(align=True)
        sub.scale_y = 0.8
        sub.enabled = False
        sub.label(text="G moves · R rotates · S scales")
        sub.label(text="Expression handles are hidden")
        sub.label(text="Save Layout locks it there for good")
    else:
        state = body.row()
        state.enabled = False
        state.label(text="Board locked - Alt+G / Alt+S cannot touch it",
                    icon='LOCKED')

    hint = body.column(align=True)
    hint.enabled = False
    hint.scale_y = 0.8
    for template, _pose_bone in handles:
        hint.label(text=f"· {boardmod.LAYOUT_LABELS.get(template, template)}")

    saved = _op_rig.board_layout(skel)
    row = body.row(align=True)
    row.operator("mhfrt.board_layout", icon='FILE_TICK',
                 text="Save Layout").action = 'SAVE'
    sub = row.row(align=True)
    sub.enabled = saved is not None
    sub.operator("mhfrt.board_layout", icon='LOOP_BACK',
                 text="Restore").action = 'RESTORE'

    row = body.row(align=True)
    row.operator("mhfrt.board_layout", icon='FILE_REFRESH',
                 text="Reset Placement").action = 'RESET'
    if saved is not None:
        forget = row.row(align=True)
        forget.alignment = 'RIGHT'
        forget.operator("mhfrt.board_layout", text="",
                        icon='X').action = 'FORGET'

    body.operator("mhfrt.board_layout", icon='MODIFIER',
                  text="Repair Handle Sizes").action = 'REPAIR'

    # The way out of a panel whose handles did not come back. Shown only when
    # there is actually something hidden, and only outside the redesign mode -
    # inside it, hidden IS the point and the box above already says so.
    if not placing:
        missing = len(boardmod.solo_hidden_bones(arm_obj))
        if missing:
            warn = body.box()
            warn.alert = True
            row = warn.row()
            row.enabled = False
            row.label(text=f"{missing} controller(s) hidden", icon='HIDE_ON')
            warn.operator("mhfrt.board_layout", icon='HIDE_OFF',
                          text="Show Controllers").action = 'REVEAL'

    note = body.row()
    note.enabled = False
    if saved is None:
        note.label(text="Nothing saved yet", icon='DOT')
    else:
        n = len(saved.get("handles") or ())
        extra = len(saved.get("bones") or ()) + len(saved.get("shape") or ())
        detail = f" + {extra} redesigned" if extra else ""
        note.label(text=f"Saved · {n} panel(s){detail} · travels in the .mhfrt",
                   icon='CHECKMARK')

    _draw_eye_target(body, mh, skel, arm_obj)
    _draw_follow_head(body, skel, arm_obj)


def _draw_eye_target(body, mh, skel, arm_obj):
    """The look-at target: fitted to the eyes, with one distance to set.

    Deliberately one slider and one button. The target's SIZE is a measurement
    of this character - its two circles belong exactly as far apart as the eye
    joints - and its position on the eye line follows from that, so neither is
    offered as something to nudge. How far out it floats is the only part that
    is a matter of taste, and it is the only part shown.
    """
    from ..core import board as boardmod
    from ..ops import op_rig as _op_rig

    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return
    if not all(name in arm_obj.pose.bones
               for name in boardmod.EYE_AIM_HANDLES):
        return

    body.separator()
    row = body.row()
    row.enabled = False
    row.label(text="Eye Target", icon='CON_TRACKTO')

    eyes = _op_rig.eye_span(skel)
    if eyes <= 0.0:
        warn = body.row()
        warn.enabled = False
        warn.label(text="No eye joints on this skeleton to measure",
                   icon='INFO')
        return

    col = body.column(align=True)
    col.prop(mh, "eye_aim_distance", text="Distance From Head")
    col.operator("mhfrt.eye_target", icon='CON_TRACKTO',
                 text="Fit Circles To Eyes").action = 'FIT'

    # The measurement itself, so "fitted" is something the artist can check
    # rather than something they have to take on trust.
    span = _op_rig.eye_aim_span(skel, arm_obj)
    note = body.row()
    note.enabled = False
    off = abs(span - eyes) * 1000.0
    if off <= 0.05:
        note.label(text=f"On the eyes · {eyes * 1000.0:.1f} mm apart",
                   icon='CHECKMARK')
    else:
        note.label(text=f"Circles {span * 1000.0:.1f} mm vs eyes "
                        f"{eyes * 1000.0:.1f} mm - off by {off:.1f} mm",
                   icon='DOT')


def _draw_follow_head(body, skel, arm_obj):
    """The board's own two follow-head switches, as one click each.

    They are bone handles on the panel - a 1 mm slider inside a frame, easy to
    miss and easy to nudge - so the same value is offered here as a toggle. The
    state shown is read off the handle itself, never off a mirror of it, because
    the handle IS the switch: an artist who drags it in the viewport sees this
    follow, and a rig exported without the add-on keeps working from it.
    """
    from ..core import board as boardmod
    from ..ops import op_rig as _op_rig
    from ..props import RIG_DESTINATION_PROP, RIG_MERGED_PROP

    if arm_obj is None or getattr(arm_obj, "pose", None) is None:
        return
    switches = [name for name in boardmod.FOLLOW_HEAD_SWITCHES
                if name in arm_obj.pose.bones]
    if not switches:
        return

    body.separator()
    head_bone = _op_rig.follow_head_bone(skel)
    row = body.row()
    row.enabled = False
    row.label(text="Follow Head", icon='CON_CHILDOF')
    if not head_bone:
        warn = body.row()
        warn.enabled = False
        warn.label(text="No head bone on this skeleton yet", icon='INFO')
        return

    col = body.column(align=True)
    for switch in switches:
        value = boardmod.follow_head_value(arm_obj, switch)
        on = value >= 0.5
        entry = col.row(align=True)
        op = entry.operator(
            "mhfrt.follow_head",
            text=boardmod.FOLLOW_HEAD_LABELS.get(switch, switch),
            icon='CHECKBOX_HLT' if on else 'CHECKBOX_DEHLT',
            depress=on)
        op.switch = switch
        op.enable = not on
        # The handle is a BLEND, and an artist who has dragged it part way
        # deserves to see that here rather than a checkbox rounding them off.
        if 0.001 < value < 0.999:
            part = entry.row(align=True)
            part.enabled = False
            part.label(text=f"{value * 100.0:.0f}%")
    note = body.row()
    note.enabled = False
    note.label(text=f"Rides '{head_bone}'", icon='BONE_DATA')

    # On a standalone rig the facial skeleton is deliberately detached from the
    # character's own armature - that is what standalone MEANS - so the bone it
    # rides is its own root, and nothing turns that when the artist turns the
    # character's head. Say it here: the switch is wired and working, there is
    # simply nothing moving underneath it, and an artist who does not know that
    # reads a working control as a broken one.
    if str(skel.get(RIG_DESTINATION_PROP, "")) == 'STANDALONE' \
            and not skel.get(RIG_MERGED_PROP):
        warn = body.column(align=True)
        warn.enabled = False
        warn.scale_y = 0.8
        warn.label(text="Standalone: nothing turns that bone but this rig",
                   icon='INFO')
        warn.label(text="Merge Into Character to ride the body's own head")


def _draw_anim(card, context, mh):
    req = flow.requirement(context, 'ANIM')
    if req:
        _locked(card, req)
        return

    from ..ops.op_anim import (face_animation_action, board_has_animation,
                               ANIM_INFO_PROP)
    from ..ops import op_rig

    skel = mh.skeleton
    anim_action = face_animation_action(skel)

    # who this clip plays on - animation is stored per character
    who = card.row()
    who.enabled = False
    who.label(text=f"Plays only on: {op_rig.character_label(skel)}",
              icon='USER')

    # -- Import -- the primary action, with the accepted formats right below
    col = card.column(align=True)
    col.scale_y = 1.35
    col.operator("mhfrt.import_face_anim", icon='IMPORT',
                 text="Import Face Animation")
    hint = card.row()
    hint.enabled = False
    hint.label(text="UE5 FBX · Live Link Face CSV · curve JSON", icon='INFO')

    # -- Loaded clip -- status + one-click remove, shown right after Import
    if anim_action is not None:
        box = card.box()
        row = box.row(align=True)
        row.label(text=skel.get(ANIM_INFO_PROP, anim_action.name),
                  icon='ACTION')
        remove = row.row(align=True)
        remove.alert = True
        remove.operator("mhfrt.remove_face_anim", text="", icon='TRASH')

    # -- Export -- saves whatever is on the board (imported clip or hand-keyed
    # polish) to a re-importable JSON; greyed out until something is keyed.
    exp = card.column(align=True)
    exp.scale_y = 1.2
    exp.enabled = anim_action is not None or board_has_animation(skel)
    exp.operator("mhfrt.export_face_anim", icon='EXPORT',
                 text="Export Face Animation")

    # -- Quick tools -- a bundled test take (the neutral reset lives with the
    # other bone settings, in the Morphs step)
    tools = card.box()
    tools.label(text="Quick Tools", icon='TOOL_SETTINGS')
    col = tools.column(align=True)
    col.scale_y = 1.1
    col.operator("mhfrt.test_face_anim", icon='FILE_TICK',
                 text="Test Animation (Bundled ROM)")
    if anim_action is None:
        tip = tools.row()
        tip.enabled = False
        tip.label(text="You can also key board controls: I > Location",
                  icon='KEY_HLT')


def _draw_export(card, context, mh):
    req = flow.requirement(context, 'EXPORT')
    if req:
        _locked(card, req)
        return

    from ..ops import op_export

    skel = mh.skeleton
    destination = str(skel.get("mhfrt_rig_destination", ""))
    # A character that arrived already rigged keeps its own armature until the
    # Merge step runs. Standalone is only "ready" when there isn't one - with a
    # body rig still separate, an FBX would carry the head and nothing else.
    others = op_export.body_rigs(context, skel)
    ready = bool(
        (destination == 'STANDALONE' and not others)
        or skel.get("mhfrt_merged_body_rig")
    )
    if not ready:
        warning = card.box()
        warning.alert = True
        warning.label(text="Merge into one armature first", icon='ERROR')
        if others:
            detail = warning.column(align=True)
            detail.enabled = False
            detail.label(text=f"'{others[0].name}' still drives the body")
            detail.label(text="A game engine needs one skeleton")
        op = warning.operator(
            "mhfrt.set_tab", icon='LOOP_BACK', text="Go to Rig")
        op.tab = 'RIG'

    unity = card.box()
    unity.label(text="Unity", icon='OUTLINER_OB_ARMATURE')
    row = unity.row()
    row.scale_y = 1.35
    row.enabled = ready
    row.operator("mhfrt.export_unity_character", icon='EXPORT',
                 text="Export Character to Unity")
    hint = unity.column(align=True)
    hint.enabled = False
    hint.label(text="One FBX · all clips")
    hint.label(text="Bones + corrective blend-shape curves")

    unreal = card.box()
    unreal.label(text="Unreal Engine 5", icon='OUTLINER_OB_ARMATURE')
    row = unreal.row()
    row.scale_y = 1.35
    row.enabled = ready
    row.operator("mhfrt.export_unreal_character", icon='EXPORT',
                 text="Export Character to Unreal Engine")
    hint = unreal.column(align=True)
    hint.enabled = False
    hint.label(text="One Skeletal Mesh FBX")
    hint.label(text="Animations folder · one FBX per clip")
    hint.label(text="Bones + corrective morph-target curves")

    blendshapes = card.box()
    blendshapes.label(text="ARKit Blend Shapes", icon='SHAPEKEY_DATA')
    row = blendshapes.row()
    row.scale_y = 1.35
    row.operator("mhfrt.generate_arkit_head", icon='EXPORT',
                 text="Export ARKit Character")
    hint = blendshapes.column(align=True)
    hint.enabled = False
    hint.label(text="Its own .blend · 52 shapes · no RigLogic needed")
    hint.label(text="Character and rig, no control board")
    hint.label(text="Sculpted correctives are baked in")

    standalone = card.box()
    standalone.label(text="Standalone Blender Rig", icon='DRIVER')
    # Two buttons rather than one: the choice between bone motion and pure
    # geometry changes what the client is handed, so it belongs here where it
    # can be read, not only inside the file browser. Either button still opens
    # on Face Motion, so the pick can be changed while choosing the path.
    choice = standalone.column(align=True)
    choice.scale_y = 1.35
    op = choice.operator("mhfrt.bake_drivers", icon='BONE_DATA',
                         text="Bake to Bones + Correctives")
    op.motion = 'BONES'
    op = choice.operator("mhfrt.bake_drivers", icon='SHAPEKEY_DATA',
                         text="Bake to Shape Keys Only")
    op.motion = 'SHAPES'
    hint = standalone.column(align=True)
    hint.enabled = False
    hint.label(text="Choose a path · writes a finished .blend")
    hint.label(text="Same control board, driven by real Blender drivers")
    hint.label(text="Bones: faithful at strong poses · heavier to play")
    hint.label(text="Shape keys: fast · facial bones retired into the")
    hint.label(text="       head deform bone, weights folded in")
    hint.label(text="Eye aim comes along · clients need no add-on")

    common = card.column(align=True)
    common.enabled = False
    common.label(text="Cage, face board and helpers are excluded", icon='INFO')
    common.label(text="Original Blender animation is restored after export")


def _draw_tune(card, context, mh):
    req = flow.requirement(context, 'TUNE')
    if req:
        _locked(card, req)
        return

    from ..ops import op_tune

    skel = mh.skeleton
    session = bool(skel and skel.get(op_tune.SESSION_ACTIVE_PROP))
    if session:
        box = card.box()
        col = box.column(align=True)
        col.label(text="Tongue Edit Active", icon='EDITMODE_HLT')
        col.label(text="Purple bones only · S scale · R rotate · G move")
        if skel.mode != 'EDIT':
            op = col.operator("mhfrt.tune_tongue", icon='LOOP_FORWARDS',
                              text="Resume Tongue Edit")
            op.action = 'RESUME'
        row = col.row(align=True)
        op = row.operator("mhfrt.tune_tongue", icon='PIVOT_CURSOR',
                          text="Reselect All · Pivot to Base")
        op.action = 'PIVOT'
        row = col.row(align=True)
        row.scale_y = 1.25
        op = row.operator("mhfrt.tune_tongue", icon='CHECKMARK', text="Finish")
        op.action = 'FINISH'
        op = row.operator("mhfrt.tune_tongue", icon='CANCEL', text="Cancel")
        op.action = 'CANCEL'
        return

    col = card.column(align=True)
    col.label(text="Position bones only - geometry and weights stay untouched.",
              icon='INFO')

    eyes = card.box()
    col = eyes.column(align=True)
    col.label(text="Eyes · exact rotation centers", icon='PIVOT_CURSOR')
    col.label(text="Hover eye: L · Shift+S · Cursor to Selected")
    hint = col.row()
    hint.enabled = False
    hint.label(text="Pivot = center of the eyeball sphere, not the mesh origin.")
    hint = col.row()
    hint.enabled = False
    hint.label(text="Verify the cursor in front + side views.")
    hint = col.row()
    hint.enabled = False
    hint.label(text="Character left = viewer's right.")

    # One row, two buttons: character right on the viewer's LEFT, character
    # left on the viewer's RIGHT - matches Parts step and how a viewer sees
    # the character in front of them.
    eye_rows = (
        ('RIGHT', "Character Right",
         rest_tuning.EYE_RIGHT_DONE_PROP, 'EYE_RIGHT'),
        ('LEFT', "Character Left",
         rest_tuning.EYE_LEFT_DONE_PROP, 'EYE_LEFT'),
    )
    row = col.row(align=True)
    row.scale_y = 1.15
    for side, label, flag, reset_feature in eye_rows:
        cell = row.column(align=True)
        entry = cell.row(align=True)
        op = entry.operator("mhfrt.place_eye_center", icon='PIVOT_CURSOR',
                            text=label)
        op.side = side
        if skel.get(flag):
            entry.label(text="", icon='CHECKMARK')
        reset = entry.row(align=True)
        reset.enabled = bool(
            skel.get(flag)
            or rest_tuning.has_manual_tuning(skel, rest_tuning.EYE_BONES[side]))
        op = reset.operator("mhfrt.reset_rest_tuning", text="",
                            icon='LOOP_BACK')
        op.feature = reset_feature

    tongue = card.box()
    col = tongue.column(align=True)
    col.label(text="Tongue · fit all purple bones", icon='BONE_DATA')
    col.label(text="Every bone comes up selected - S, R, G move them together.")
    hint = col.row()
    hint.enabled = False
    hint.label(text="Use Alt+Z if the tongue is hidden inside the mouth.")
    row = col.row(align=True)
    row.scale_y = 1.25
    op = row.operator("mhfrt.tune_tongue", icon='EDITMODE_HLT',
                      text="Edit Tongue Bones")
    op.action = 'START'
    if skel.get(rest_tuning.TONGUE_DONE_PROP):
        row.label(text="", icon='CHECKMARK')
    reset = row.row(align=True)
    reset.enabled = bool(
        skel.get(rest_tuning.TONGUE_DONE_PROP)
        or rest_tuning.has_manual_tuning(skel, rest_tuning.TONGUE_BONES))
    op = reset.operator("mhfrt.reset_rest_tuning", text="", icon='LOOP_BACK')
    op.feature = 'TONGUE'
    keys = col.row(align=True)
    keys.enabled = False
    keys.label(text="S  Scale")
    keys.label(text="R  Rotate")
    keys.label(text="G  Move")

    col = card.column()
    col.scale_y = 1.35
    col.operator("mhfrt.finish_rest_tuning", icon='CHECKMARK',
                 text="Confirm Positions & Continue")


def _draw_bind(card, context, mh):
    req = flow.requirement(context, 'BIND')
    if req:
        _locked(card, req)
        return
    bound = flow.bind_done(mh)

    col = card.column()
    col.scale_y = 1.4
    col.operator("mhfrt.transfer_weights", icon='MOD_VERTEX_WEIGHT',
                 text="Re-Bind" if bound else "Transfer Weights & Bind")

    if bound:
        row = card.row()
        row.label(text="Character is live - use its board", icon='CHECKMARK')

    body = _section(card, mh, "ui_sec_cleanup", "Weight Cleanup",
                    icon_value=icons.icon("polish"))
    if body:
        head = body.row(align=True)
        sub = head.row()
        sub.enabled = False
        sub.label(text="Optional - for sealed lips / eyelids")
        _help_btn(head, 'CLEANUP')

        from ..ops.op_wrap import MOUTH_OPEN_KEY, CLOSE_EYES_KEY
        cage = mh.cage
        keys = (cage.data.shape_keys.key_blocks
                if (cage and cage.data.shape_keys) else None)
        wrapped = flow.wrap_done(mh)
        mouth_ok = bool(wrapped and keys and MOUTH_OPEN_KEY in keys)
        eyes_ok = bool(wrapped and keys and CLOSE_EYES_KEY in keys)

        # one slider per feature: cage AND head pose together, the head's
        # pose key is generated automatically from the wrapped cage. Dragging
        # them costs nothing - Re-Bind applies them behind the scenes.
        col = body.column(align=True)
        row = col.row(align=True)
        row.enabled = mouth_ok
        row.prop(mh, "mouth_open_amount", slider=True)
        row = col.row(align=True)
        row.enabled = eyes_ok
        row.prop(mh, "eyes_close_amount", slider=True)

        if mouth_ok or eyes_ok:
            hint = body.column(align=True)
            hint.enabled = False
            hint.label(text="Set sliders, then Re-Bind above", icon='INFO')
            hint.label(text="It poses both meshes for you", icon='BLANK1')
            # Only needed to SEE the pose (to sculpt or smooth-brush it). The
            # bind never depends on it.
            row = body.row(align=True)
            row.scale_y = 1.15
            row.operator("mhfrt.apply_cleanup_pose", icon='SHAPEKEY_DATA',
                         text="Pose Meshes To Sliders")
            hint = body.row()
            hint.enabled = False
            hint.label(text="Optional · to sculpt in the pose", icon='BLANK1')
        else:
            hint = body.row()
            hint.enabled = False
            hint.label(text="Needs the wrapped bundled cage (guide keys)",
                       icon='INFO')


def _picker_live(context):
    """The live-ranked picker is worth building only when a pose is being
    held. During playback the rig re-evaluates every frame, so ranking hundreds
    of channels would run each frame and the list would reshuffle too fast to
    click - freeze it while playing."""
    screen = getattr(context, "screen", None)
    return not (screen is not None and screen.is_animation_playing)


_MORPH_LIST_LOOKUP = {}


def _morph_list_cache_key(context, obj):
    return (context.scene.as_pointer(), obj.as_pointer() if obj else 0)


# What a morph object IS, shown ahead of its name.  A mesh attached in the
# Parts step reads as that part rather than as a bare object name: with eight
# `shirt.0xx` meshes in the list, "Lash Upper Right · shirt.015" is the only
# way to know which one you are about to sculpt.
_PART_ICONS = {
    'EYE_L': 'HIDE_OFF', 'EYE_R': 'HIDE_OFF',
    'TEETH_UPPER': 'RIGID_BODY', 'TEETH_LOWER': 'RIGID_BODY',
    'LASH_UPPER_R': 'CURVE_DATA', 'LASH_UPPER_L': 'CURVE_DATA',
    'LASH_LOWER_R': 'CURVE_DATA', 'LASH_LOWER_L': 'CURVE_DATA',
    'TONGUE': 'BONE_DATA',
}


def _morph_object_role(mh, obj, index):
    """(label, icon) for one Objects row: Head, its Parts slot, or the cage."""
    if index == 0:
        return f"Head · {obj.name}", 'USER'
    from ..ops import op_attach
    part = obj.get(op_attach.ATTACH_PART_PROP)
    if part:
        return (f"{op_attach.part_label(part)} · {obj.name}",
                _PART_ICONS.get(part, 'MESH_DATA'))
    if mh.cage is not None and obj == mh.cage:
        return f"Cage · {obj.name}", 'MOD_MESHDEFORM'
    return obj.name, 'MESH_DATA'


class MHFRT_UL_morph_objects(bpy.types.UIList):
    """Head + extra targets in a native Blender selectable list."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        obj = item.obj
        if obj is None:
            layout.label(text="Missing Object", icon='ERROR')
            return
        label, row_icon = _morph_object_role(data, obj, index)
        op = layout.operator(
            "mhfrt.set_active_morph_object",
            text=label,
            icon=row_icon,
            emboss=False,
        )
        op.index = index
        op.object_name = obj.name


class MHFRT_UL_morphs(bpy.types.UIList):
    """Pose-filtered DNA channels, kept in DNA order, with lazy key creation."""

    def filter_items(self, context, data, propname):
        from ..ops import op_morphs

        rows = getattr(data, propname)
        obj = op_morphs.active_morph_object(data, context)
        active = op_morphs.picker_items(data, obj) if obj else []
        lookup = {name: (value, exists)
                  for name, value, exists in active}
        cache_key = _morph_list_cache_key(context, obj)
        scene_pointer = cache_key[0]
        for old_key in tuple(_MORPH_LIST_LOOKUP):
            if old_key[0] == scene_pointer and old_key != cache_key:
                del _MORPH_LIST_LOOKUP[old_key]
        _MORPH_LIST_LOOKUP[cache_key] = lookup

        only_muted = bool(getattr(data, "morph_show_muted_only", False))
        muted = op_morphs.preview_muted(data.skeleton) if only_muted else ()
        flags = []
        for item in rows:
            show = item.key_name in lookup
            if show and only_muted:
                show = op_morphs.riglogic.morph_source_name(
                    item.key_name) in muted
            flags.append(self.bitflag_filter_item if show else 0)

        # Ordered by how strongly the pose fires each morph, then by name so
        # equal values have a stable, readable order.  The value is the one the
        # RigLogic inputs report BEFORE any bone dialling or preview mute, so
        # silencing a row's bones never moves it - it stays exactly where the
        # artist last saw it (see op_morphs.rig_input_values).
        order = sorted(
            range(len(rows)),
            key=lambda i: (-lookup.get(rows[i].key_name, (0.0, False))[0],
                           rows[i].key_name.lower()))
        # new_order maps collection index -> display position.
        new_order = [0] * len(rows)
        for position, index in enumerate(order):
            new_order[index] = position
        return flags, new_order

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        from ..ops import op_morphs

        obj = op_morphs.active_morph_object(data, context)
        if obj is None:
            layout.label(text=item.key_name)
            return
        lookup = _MORPH_LIST_LOOKUP.get(
            _morph_list_cache_key(context, obj), {})
        value, _exists = lookup.get(item.key_name, (0.0, False))
        kb = op_morphs.existing_morph_key(obj, item.key_name)

        row = layout.row(align=True)
        op = row.operator("mhfrt.pick_morph",
                          text=f"{item.key_name}  ·  {value:.2f}",
                          icon='SHAPEKEY_DATA', emboss=False)
        op.key_name = item.key_name
        op.object_name = obj.name
        op.action = 'SELECT'
        op = row.operator("mhfrt.pick_morph", text="",
                          icon='SCULPTMODE_HLT', emboss=False)
        op.key_name = item.key_name
        op.object_name = obj.name
        op.action = 'SCULPT'
        # Preview this morph without its bones, so its sculpted shape can be
        # judged on its own.  This is a VIEWING state: it does not touch the
        # channel's value or its Channels Intensity, and the row keeps its
        # place in the list.  The Channels Intensity slider under the list is
        # the real setting and is shown separately below.
        source = op_morphs.riglogic.morph_source_name(item.key_name)
        if op_morphs.channel_drives_bones(source):
            previewing = op_morphs.is_preview_muted(data.skeleton, source)
            isolate = row.operator(
                "mhfrt.toggle_morph_isolate", text="",
                icon='HIDE_ON' if previewing else 'BONE_DATA',
                emboss=False, depress=previewing,
            )
            isolate.key_name = item.key_name
        if kb is not None:
            is_solo = bool(obj.show_only_shape_key
                           and obj.active_shape_key == kb)
            op = row.operator(
                "mhfrt.select_morph_key", text="",
                icon='SOLO_ON' if is_solo else 'SOLO_OFF',
                emboss=False,
            )
            op.key_name = item.key_name
            op.object_name = obj.name
            op.action = 'SOLO'
            row.prop(kb, "mute", text="", emboss=False,
                     icon='CHECKBOX_DEHLT' if kb.mute else 'CHECKBOX_HLT')


def _draw_parts(card, context, mh):
    req = flow.requirement(context, 'PARTS')
    if req:
        _locked(card, req)
        return

    from ..ops import op_attach

    col = card.column(align=True)
    col.label(text="Select your mesh, then press its slot.", icon='INFO')

    attached = {}
    for obj in op_attach.attached_objects(mh):
        attached.setdefault(obj.get(op_attach.ATTACH_PART_PROP), []).append(obj)

    def remove(row, part):
        """The X that hands this slot's meshes back. Only when it HAS any."""
        cell = row.row(align=True)
        cell.alert = True
        op = cell.operator("mhfrt.detach_part", text="", icon='X')
        op.part = part
        op.object_name = ""

    def slot(layout, part, text=None):
        row = layout.row(align=True)
        row.scale_y = 1.2
        op = row.operator("mhfrt.attach_part", icon='GROUP_BONE',
                          text=text or op_attach.PART_LABELS[part])
        op.part = part
        op.custom_name = ""
        if attached.get(part):
            row.label(text="", icon='CHECKMARK')
            remove(row, part)

    def custom_slot(layout, part):
        """An existing custom slot, re-attachable exactly like a built-in one."""
        name = op_attach.custom_part_name(part)
        row = layout.row(align=True)
        row.scale_y = 1.2
        op = row.operator("mhfrt.attach_part", icon='GROUP_BONE', text=name)
        op.part = 'CUSTOM'
        op.custom_name = name
        if attached.get(part):
            row.label(text="", icon='CHECKMARK')
        # A custom slot exists only while something is in it, so its X is
        # always offered: it is the only way to get rid of one, where a
        # built-in slot merely empties.
        remove(row, part)

    box = card.box()
    col = box.column(align=True)
    col.label(text="Eyes · one bone, 100%", icon='HIDE_OFF')
    # character right sits on the viewer's left, matching the viewport
    row = col.row(align=True)
    slot(row, 'EYE_R')
    slot(row, 'EYE_L')

    box = card.box()
    col = box.column(align=True)
    col.label(text="Teeth · one bone, 100%", icon='RIGID_BODY')
    row = col.row(align=True)
    slot(row, 'TEETH_UPPER')
    slot(row, 'TEETH_LOWER')

    box = card.box()
    col = box.column(align=True)
    col.label(text="Eyelashes · topology weights from head", icon='CURVE_DATA')
    row = col.row(align=True)
    slot(row, 'LASH_UPPER_R', text="Upper Right")
    slot(row, 'LASH_UPPER_L', text="Upper Left")
    row = col.row(align=True)
    slot(row, 'LASH_LOWER_R', text="Lower Right")
    slot(row, 'LASH_LOWER_L', text="Lower Left")

    box = card.box()
    col = box.column(align=True)
    col.label(text="Tongue · automatic weights", icon='BONE_DATA')
    slot(col, 'TONGUE')

    # Slots the artist names themselves - brows, fuzz, stubble, a piercing.
    # Bound the eyelash way, because that is what works for any mesh sitting on
    # the face with no bone of its own.
    customs = op_attach.custom_parts(mh)
    box = card.box()
    col = box.column(align=True)
    col.label(text="Custom · topology weights from head", icon='MESH_DATA')
    for part in customs:
        custom_slot(col, part)
    row = col.row(align=True)
    row.prop(mh, "part_custom_name", text="", icon='GREASEPENCIL')
    typed = " ".join(mh.part_custom_name.split())
    add = row.row(align=True)
    add.enabled = bool(typed)
    op = add.operator("mhfrt.attach_part", icon='ADD',
                      text="Add" if typed not in
                      {op_attach.custom_part_name(p) for p in customs}
                      else "Attach")
    op.part = 'CUSTOM'
    op.custom_name = typed

    if attached:
        listed = [part for part, _label, _bone in op_attach.PART_SPECS]
        box = card.box()
        col = box.column(align=True)
        for part in listed + [p for p in attached if p not in listed]:
            for obj in attached.get(part, ()):
                row = col.row(align=True)
                name = row.row(align=True)
                name.enabled = False
                name.label(text=f"{op_attach.part_label(part)}   {obj.name}",
                           icon='CHECKMARK')
                # Per-MESH removal, for a slot holding more than one
                drop = row.row(align=True)
                drop.alert = True
                op = drop.operator("mhfrt.detach_part", text="", icon='X')
                op.part = part
                op.object_name = obj.name


def _draw_morphs(card, context, mh):
    req = flow.requirement(context, 'MORPHS')
    if req:
        _locked(card, req)
        return

    from ..ops import op_morphs

    active_obj = op_morphs.active_morph_object(mh, context)
    # The last mesh clicked in the viewport, when it is not one of this
    # character's morph objects.  While one is on screen the Morphs list is
    # replaced by the warning below: the artist is looking at a mesh the step
    # cannot sculpt, and offering a Sculpt button that would silently jump to
    # the head instead is exactly the mix-up this prevents.
    stray = op_morphs.unregistered_mesh(mh, context)
    # -1 is a normal state: no row selected, because the pose fires none.
    selected_index = int(mh.morph_ui_active_index)
    selected_source = (mh.morph_ui_items[selected_index].key_name
                       if 0 <= selected_index < len(mh.morph_ui_items) else "")
    listed = None       # rows the list drew, once template_list has run

    box = card.box()
    box.label(text="Objects", icon='OUTLINER_OB_MESH')
    row = box.row()
    row.template_list(
        "MHFRT_UL_morph_objects", "",
        mh, "morph_object_ui_items",
        mh, "morph_object_ui_active_index",
        rows=3,
    )
    buttons = row.column(align=True)
    buttons.operator("mhfrt.add_morph_objects", text="", icon='ADD')
    remove = buttons.column(align=True)
    object_index = int(mh.morph_object_ui_active_index)
    remove.enabled = 0 < object_index < len(mh.morph_object_ui_items)
    op = remove.operator("mhfrt.remove_morph_object", text="", icon='REMOVE')
    if remove.enabled:
        item_obj = mh.morph_object_ui_items[object_index].obj
        op.object_name = item_obj.name if item_obj else ""
    hint = box.row()
    hint.enabled = False
    hint.label(text="Select meshes, then + · Head cannot be removed", icon='INFO')

    if stray is not None:
        # Name it the same way the list would, so a mesh attached in the Parts
        # step is recognisable ("Tongue · shirt") and not just an object name.
        stray_label, _icon = _morph_object_role(mh, stray, -1)
        box = card.box()
        box.alert = True
        col = box.column(align=True)
        col.label(text=f"'{stray_label}' is not in this list", icon='ERROR')
        col.label(text="Nothing here would sculpt it. Add it, or pick a mesh")
        col.label(text="from the list above - clicking a row selects it.")
        row = box.row(align=True)
        row.scale_y = 1.2
        op = row.operator("mhfrt.add_morph_objects",
                          text=f"Add '{stray.name}'", icon='ADD')
        op.object_name = stray.name
        if active_obj is not None:
            op = row.operator("mhfrt.set_active_morph_object",
                              text=f"Back to {active_obj.name}",
                              icon='RESTRICT_SELECT_OFF')
            op.object_name = active_obj.name
            op.index = op_morphs.morph_object_index(mh, active_obj)
    elif active_obj:
        box = card.box()
        head = box.row()
        active_label, _icon = _morph_object_role(
            mh, active_obj, op_morphs.morph_object_index(mh, active_obj))
        head.label(text=f"Morphs · {active_label}", icon='SHAPEKEY_DATA')
        if not _picker_live(context):
            row = box.row()
            row.enabled = False
            row.label(text="Active morphs resume when playback stops",
                      icon='PLAY')
        else:
            # Threshold slider: only morphs at or above this value show up
            # in the list.  Default 0.35 keeps the list focused on morphs
            # meaningfully firing on the current pose.  The second toggle
            # narrows the list to the channels currently previewed without
            # their bones, so they can be found and switched back without
            # hunting for them.
            previews = op_morphs.preview_muted(mh.skeleton)
            hdr = box.row(align=True)
            hdr.prop(mh, "morph_display_threshold", slider=True,
                     text="Show Above")
            sub = hdr.row(align=True)
            sub.enabled = bool(previews)
            sub.prop(mh, "morph_show_muted_only",
                     text=f"No Bones ({len(previews)})" if previews
                     else "No Bones",
                     toggle=True, icon='HIDE_ON')
            row = box.row()
            row.template_list(
                "MHFRT_UL_morphs", "",
                mh, "morph_ui_items",
                mh, "morph_ui_active_index",
                rows=6,
            )
            buttons = row.column(align=True)
            selected_key = op_morphs.existing_morph_key(
                active_obj, selected_source)
            one = buttons.column(align=True)
            one.enabled = selected_key is not None
            remove_selected = one.operator(
                "mhfrt.remove_morph_key", text="", icon='REMOVE')
            remove_selected.source_name = selected_source
            remove_selected.object_name = active_obj.name
            # Housekeeping on the whole object, under the per-row minus: both
            # delete driven keys, so they belong on the same column - and
            # neither depends on a row being selected.
            buttons.separator()
            op = buttons.operator("mhfrt.manage_morph_keys", text="",
                                  icon='MOD_DECIM')
            op.action = 'COMPACT'
            danger = buttons.column(align=True)
            danger.alert = True
            op = danger.operator("mhfrt.manage_morph_keys", text="",
                                 icon='TRASH')
            op.action = 'REMOVE_ALL'
            # template_list ran the UIList's filter above, so the row set it
            # just built tells us whether the list came out empty - a neutral
            # pose fires nothing, and that empty state is normal.
            listed = _MORPH_LIST_LOOKUP.get(
                _morph_list_cache_key(context, active_obj))
            hint = box.row()
            hint.enabled = False
            if listed is not None and not listed:
                hint.label(text="Neutral pose · pose a control to fill the list",
                           icon='INFO')
            else:
                hint.label(text="Pose controls · strongest first, then by name",
                           icon='INFO')
            if previews:
                back = box.row(align=True)
                back.operator("mhfrt.clear_morph_previews",
                              icon='LOOP_BACK',
                              text=f"Bring Bones Back ({len(previews)})")

    # Bone strength - the whole face, then the one channel the list is on.
    # Dialling a channel to 0 leaves its shape key deforming, so it runs on
    # its sculpted shape alone.  The neutral-pose reset is NOT here: it moves
    # board controls, so it lives with the board button in the Rig list.
    box = card.box()
    box.label(text="Bones Settings", icon='BONE_DATA')
    box.prop(mh, "riglogic_scale_mul", text="Expressions Intensity",
             slider=True)

    # Channels Intensity is about ONE channel, so it only exists while a row is
    # selected - no selection is the normal state on a neutral pose, and a
    # placeholder asking for a row that isn't there is just noise.
    col = box.column(align=True)
    if selected_source:
        box.separator()
        box.label(text="Channels Intensity")
        col = box.column(align=True)
        if not op_morphs.channel_drives_bones(selected_source):
            row = col.row()
            row.enabled = False
            row.label(text=f"{selected_source} · no board channel of its own",
                      icon='INFO')
        else:
            label = col.row()
            label.enabled = False
            label.label(text=f"Channel · {selected_source}")
            col.prop(mh, "morph_channel_intensity", text="This Channel",
                     slider=True)
            hint = col.row()
            hint.enabled = False
            hint.label(
                text="0 = this channel really drives no bones (a rig setting)",
                icon='INFO')
    # The two summaries below are about the character, not the selected row, so
    # they stay whatever the list is doing - each carries the way back.
    dialled = op_morphs.channel_weights(mh.skeleton)
    if dialled:
        row = col.row(align=True)
        row.label(text=f"{len(dialled)} channel(s) dialled down", icon='BONE_DATA')
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        sub.operator("mhfrt.restore_morph_channels", text="", icon='LOOP_BACK')
    previewing = op_morphs.preview_muted(mh.skeleton)
    if previewing:
        row = col.row(align=True)
        row.label(text=f"{len(previewing)} previewed without bones",
                  icon='HIDE_ON')
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        sub.operator("mhfrt.clear_morph_previews", text="", icon='LOOP_BACK')


_TAB_DRAW = {
    'SETUP': _draw_setup,
    'LANDMARKS': _draw_landmarks,
    'WRAP': _draw_wrap,
    'RIG': _draw_rig,
    'TUNE': _draw_tune,
    'BIND': _draw_bind,
    'PARTS': _draw_parts,
    'MORPHS': _draw_morphs,
    'ANIM': _draw_anim,
    'EXPORT': _draw_export,
}


# ----------------------------------------------------------------- panel ---

class MHFRT_PT_main(bpy.types.Panel):
    bl_label = "MetaHuman Rig Transfer"
    bl_idname = "MHFRT_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MH Transfer"

    def draw_header(self, context):
        self.layout.label(text="", icon_value=icons.icon("logo"))

    def draw(self, context):
        layout = self.layout
        mh = context.scene.mhfrt
        steps = flow.steps(context)
        tab = mh.ui_tab
        idx = flow.STEP_IDS.index(tab)
        done_n = sum(s.done for s in steps)

        # -- character list: who the whole panel is working on --
        _draw_characters(layout, context, mh)

        # -- workflow strip: icon tabs + progress --
        box = layout.box()
        col = box.column(align=True)
        row = col.row(align=True)
        row.scale_y = 1.25
        for s in steps:
            op = row.operator("mhfrt.set_tab", text="",
                              depress=(s.id == tab),
                              **_tab_icon_kwargs(s.id, s.done))
            op.tab = s.id
        sub = row.row(align=True)
        sub.emboss = 'NONE'
        sub.operator("mhfrt.help", text="", icon='QUESTION').topic = 'OVERVIEW'
        col.progress(factor=done_n / len(steps), type='BAR',
                     text="Workflow complete" if done_n == len(steps)
                     else f"{done_n} of {len(steps)} steps done")

        # -- display bar: always available, every step --
        col.separator(factor=0.4)
        row = col.row(align=True)
        row.prop(mh, "view_mode", expand=True, icon_only=True)
        row.separator()
        row.prop(mh, "studio_shading", text="", icon='SHADING_SOLID')
        row.prop(mh, "cage_studio", text="", icon='MOD_WIREFRAME')
        row.prop(mh, "cage_in_front", text="", icon='XRAY')
        row.prop(mh, "show_pairs_overlay", text="", icon='PMARKER_ACT')
        sub = row.row(align=True)
        sub.enabled = mh.skeleton is not None
        sub.prop(mh, "show_bones", text="", icon='BONE_DATA')

        # -- active step card --
        card = layout.box()
        head = card.row(align=True)
        head.label(text=f"{idx + 1}  ·  {steps[idx].label}",
                   **_tab_icon_kwargs(tab, steps[idx].done))
        _help_btn(head, tab)
        _TAB_DRAW[tab](card, context, mh)

        # -- prev / next footer --
        row = layout.row(align=True)
        if idx > 0:
            op = row.operator("mhfrt.set_tab", text="", icon='TRIA_LEFT')
            op.tab = flow.STEP_IDS[idx - 1]
        if idx + 1 < len(steps):
            sub = row.row(align=True)
            # Optional steps (parts / morphs / anim) are skippable: keep Next
            # always active so a character without teeth or morphs can walk
            # straight through them.
            sub.active = steps[idx].done or tab in flow.OPTIONAL_STEPS
            op = sub.operator("mhfrt.set_tab", icon='TRIA_RIGHT',
                              text=f"Next  ·  {steps[idx + 1].label}")
            op.tab = flow.STEP_IDS[idx + 1]
        elif done_n == len(steps):
            row.label(text="All steps complete", icon='CHECKMARK')


_classes = (
    MHFRT_OT_set_tab,
    MHFRT_OT_help,
    MHFRT_UL_rigs,
    MHFRT_UL_morph_objects,
    MHFRT_UL_morphs,
    MHFRT_PT_main,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
    _MORPH_LIST_LOOKUP.clear()
    icons.unregister()
