@tool
extends EditorScenePostImport
## Assign as the GLB's import script to enable exported diffuse vertex colors.

func _post_import(scene: Node) -> Object:
    _enable_vertex_colors(scene)
    return scene

func _enable_vertex_colors(node: Node) -> void:
    if node is MeshInstance3D:
        var instance := node as MeshInstance3D
        if instance.mesh != null:
            for surface in range(instance.mesh.get_surface_count()):
                var arrays := instance.mesh.surface_get_arrays(surface)
                var colors = arrays[Mesh.ARRAY_COLOR]
                if colors == null or colors.is_empty():
                    continue
                var material := instance.get_active_material(surface)
                if material is BaseMaterial3D:
                    var colored_material := material.duplicate() as BaseMaterial3D
                    colored_material.vertex_color_use_as_albedo = true
                    instance.set_surface_override_material(surface, colored_material)
    for child in node.get_children():
        _enable_vertex_colors(child)
