//! Turns a `world::World` into Bevy entities.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use world::{Heightmap, ObstacleShape, World as SimWorld};

/// Any entity created from the world, removed at each rebuild.
#[derive(Component)]
pub struct WorldEntity;

pub fn spawn_world(
    world: &SimWorld,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(terrain_mesh(&world.terrain))),
        MeshMaterial3d(materials.add(StandardMaterial {
            perceptual_roughness: 0.95,
            ..default()
        })),
        WorldEntity,
    ));

    let obstacle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.52, 0.5),
        perceptual_roughness: 0.8,
        ..default()
    });
    for o in &world.obstacles {
        spawn_obstacle(o, obstacle_mat.clone(), commands, meshes);
    }

    // Marker of the drone's starting point (origin).
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(1.0, 0.05))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.95, 0.45, 0.1),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, world.terrain.height_at(0.0, 0.0) + 0.05, 0.0),
        WorldEntity,
    ));
}

/// A world obstacle, as a Bevy entity.
pub fn spawn_obstacle(
    o: &world::Obstacle,
    material: Handle<StandardMaterial>,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
) {
    let mesh = match o.shape {
        ObstacleShape::Box { size } => meshes.add(Cuboid::from_size(size)),
        ObstacleShape::Cylinder { radius, height } => meshes.add(Cylinder::new(radius, height)),
    };
    // Bevy primitives are centered: raise by half the height.
    let center = o.base + Vec3::Y * (o.height() / 2.0);
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(center).with_rotation(Quat::from_rotation_y(o.yaw)),
        WorldEntity,
    ));
}

/// Triangulated mesh of the height grid, colored by altitude.
fn terrain_mesh(t: &Heightmap) -> Mesh {
    let n = t.vertices_per_side();
    let (lo, hi) = t.range();
    let span = (hi - lo).max(1e-3);

    let mut positions = Vec::with_capacity(n * n);
    let mut colors = Vec::with_capacity(n * n);
    let mut uvs = Vec::with_capacity(n * n);
    for iz in 0..n {
        for ix in 0..n {
            let (x, z) = t.vertex_xz(ix, iz);
            let y = t.heights()[iz * n + ix];
            positions.push([x, y, z]);
            uvs.push([ix as f32 / (n - 1) as f32, iz as f32 / (n - 1) as f32]);
            // Green at the bottom, grey-brown at the top.
            let k = (y - lo) / span;
            let low = Vec3::new(0.30, 0.48, 0.22);
            let high = Vec3::new(0.55, 0.50, 0.42);
            let c = low.lerp(high, k);
            colors.push([c.x, c.y, c.z, 1.0]);
        }
    }

    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for iz in 0..n - 1 {
        for ix in 0..n - 1 {
            let i = (iz * n + ix) as u32;
            let (right, down) = (i + 1, i + n as u32);
            // Counter-clockwise order seen from above (+Y): visible faces.
            indices.extend_from_slice(&[i, down, right, right, down, down + 1]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}
