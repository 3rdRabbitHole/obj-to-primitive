# Object to Primitive

Blender addon that classifies mesh objects as **BOX**, **CYLINDER**, **SPHERE**, or **UNKNOWN** and replaces them with simple geometric primitives.

Useful for reducing high-poly meshes (e.g. CAD imports) into lightweight proxy geometry.

## Features

- **PCA-based classification** — eigenvalue analysis with circularity testing and face normal spread
- **Preview before commit** — see the primitive replacement before applying
- **Convex hull fallback** — UNKNOWN shapes get a convex hull instead of a primitive
- **Batch processing** — select multiple objects, preview and confirm all at once
- **Original preservation** — originals are moved to an archive collection, not deleted
- **Customizable colors** — change the preview color for each shape type in the Primitive Color panel
- **Experimental classifiers** — RANSAC and Hybrid methods available for testing

## Requirements

- Blender 4.2 or later
- NumPy (bundled with Blender)

## Installation

1. Download the latest release zip from [Releases](../../releases)
2. In Blender: **Edit → Preferences → Add-ons → Install...**
3. Select the downloaded `obj_to_primitive.zip`
4. Enable "Object to Primitive" in the addon list

Or manually copy the `obj_to_primitive/` folder into your Blender addons directory.

## Usage

1. Select one or more mesh objects
2. Open the sidebar (**N**) → **Obj2Prim** tab
3. Adjust settings if needed (algorithm, cylinder segments, etc.)
4. Click **Preview** to see the primitive replacements
5. During preview, select any primitive and click **Exclude Selected** to remove it from conversion
6. Click **Confirm** to apply, or **Cancel** to revert

### Viewing preview colors

Preview primitives are colored by shape type using **Object Color**. To see the colors in the viewport:

- **Solid mode**: set Viewport Shading color to **Object** (header shading dropdown → Color → Object)
- **Material Preview / Rendered mode**: colors are applied via material and visible by default

You can customize the colors in the **Primitive Color** panel (collapsed by default).

## Notes

- **RANSAC classifier**: marked experimental. It may produce more accurate rotation on very high-poly objects with non-uniform vertex distribution (e.g. dense CAD meshes), but lacks some refinements of the PCA method. Use PCA for general cases.
- **Hybrid classifier**: runs PCA first, falls back to RANSAC when PCA confidence is low. Inherits RANSAC limitations.

## License

[MIT](LICENSE)
