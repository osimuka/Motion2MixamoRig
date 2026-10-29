# Import an exported animation into Godot 4

The existing `mixamo_character.glb` is glTF 2.0 and works in Godot 4; no
separate Godot export or FBX conversion is needed. It contains the character,
skin weights, skeleton, and an animation named `mixamo_clip`.

## Import and play

1. Extract the downloaded ZIP and copy `mixamo_character.glb` into your
   Godot project's filesystem, for example `res://characters/`.
2. Wait for import, then drag the GLB from the FileSystem dock into a 3D scene.
3. Open an inherited scene if you need to inspect or customize the imported
   nodes. The imported character contains a `Skeleton3D`, skinned mesh nodes,
   and an `AnimationPlayer`.
4. Select `AnimationPlayer`, choose `mixamo_clip`, and press Play to preview.
   To play in-game, call `play("mixamo_clip")` on that player:

   ```gdscript
   @onready var player: AnimationPlayer = $Character/AnimationPlayer

   func _ready() -> void:
       player.play("mixamo_clip")
   ```

   Adjust `$Character/AnimationPlayer` to match your scene tree.

The exported clip is not automatically looped: a recorded action need not have
matching first and last poses. Enable looping in Godot's advanced animation
import settings only when the motion is suitable, then reimport.

## Preserve the mannequin's blue colors

The exporter stores diffuse-sampled colors in glTF `COLOR_0`. Some Godot
imports retain the color arrays but leave the material's **Vertex Color →
Use as Albedo** disabled. If the character looks white, enable that property
on its material, or apply the supplied import helper:

1. Copy [`godot_post_import.gd`](../scripts/godot_post_import.gd) into the Godot
   project, for example `res://tools/godot_post_import.gd`.
2. Select the GLB in the FileSystem dock. In the Import dock, set its
   **Import Script** to that script and select **Reimport**. Depending on the
   editor version, the setting may be inside advanced import settings.
3. The helper enables vertex colors on imported meshes that actually have
   color arrays, without replacing their skinning or animation.

The output uses sampled diffuse colors, not the original normal/roughness
texture maps. Those source PBR maps are not embedded in this GLB.

## Movement and reuse

- Units are metres and the GLB is Y-up; Godot handles the import axes.
- Hip translation is part of the animation. Playing it moves the skeleton
  inside the character scene, not a `CharacterBody3D` collision controller.
  Configure root motion in an `AnimationTree` and apply it to your controller
  if your game needs physics-driven movement.
- This is a skinned character-and-animation export, not automatic retargeting
  to every Godot character. To reuse it on a different humanoid, configure
  Godot's humanoid bone mapping/retargeting, or generate again using that
  character's compatible FBX.
- Inspect `diagnostics.json` and the visible motion before shipping. Import
  compatibility does not guarantee foot contact or motion quality.

## Validation

A generated Default Mannequin GLB was imported through the Godot 4.7.2 editor
and instantiated headlessly. It produced one 55-bone `Skeleton3D`, one skinned
mesh, and an `AnimationPlayer` with `mixamo_clip`. Seeking the clip changed
bone rotations. Vertex color arrays were retained; reimporting with the
material helper enabled the albedo flag while preserving working animation.
Other Godot versions and arbitrary
character rigs have not all been tested.
