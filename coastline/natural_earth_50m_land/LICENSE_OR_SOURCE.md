# Natural Earth 50m Land — Source and Licence

## Data provenance

| Field | Value |
|---|---|
| Dataset | Natural Earth – 50m Physical Vectors: Land |
| Resolution | 1:50 million |
| Source repository | https://github.com/nvkelso/natural-earth-vector |
| Branch at retrieval | `master` |
| Git commit SHA at retrieval | *Not recorded at download time — see note below* |
| Official distribution page | https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-land/ |
| Retrieved | 2026-09-23 |

### Note on commit SHA

The exact commit SHA of `nvkelso/natural-earth-vector` at the moment the files
were downloaded on 2026-09-23 was not captured.  The SHA-256 checksums below
serve as the definitive fingerprint of the bundled files.  To verify which
commit they correspond to, compare the checksums against the GitHub history:

```
https://github.com/nvkelso/natural-earth-vector/commits/master/50m_physical/ne_50m_land.shp
```

If the checksums match a specific commit's tree, that is the provenance commit.
Alternatively, retrieve the files at a known tagged release (Natural Earth
periodically issues versioned releases) and compare checksums.

## Raw GitHub URLs used for retrieval

Files were downloaded from the master branch of the canonical repository.
These URLs resolve to the **current** master head, not to the specific commit
used; pin to a commit SHA in the URL to reproduce the exact retrieval:

| File | Raw URL (master HEAD — use commit SHA for exact reproduction) |
|---|---|
| `ne_50m_land.shp` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.shp |
| `ne_50m_land.shx` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.shx |
| `ne_50m_land.dbf` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.dbf |
| `ne_50m_land.prj` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.prj |
| `ne_50m_land.cpg` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.cpg |

## Files bundled and SHA-256 checksums

These checksums uniquely identify the files committed to this repository.

| File | Size (approx.) | SHA-256 |
|---|---|---|
| `ne_50m_land.shp` | 1.1 MB | `140d1419f7ace66bf0ea8ceb068cab544d413170f4e8c48e0a11bde6c978b319` |
| `ne_50m_land.shx` | 12 KB | `c1b93ee40fe83d6322197251517c141c2c91970915ac61d7bf7a4821fc2a68d7` |
| `ne_50m_land.dbf` | 72 KB | `d082add78c82e01b9bf422df3143ad6fbe43ca1bcb798d12a334f9e001b67bc0` |
| `ne_50m_land.prj` | 4 KB | `98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb` |
| `ne_50m_land.cpg` | <1 KB | `3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824` |

Total features: 1,420 polygons (Polygon type, WGS 84 / PlateCarree).

To verify integrity of the bundled files:

```bash
sha256sum coastline/natural_earth_50m_land/ne_50m_land.*
```

## Licence

Natural Earth data is in the **public domain**.

> Natural Earth is a public domain map dataset available at 1:10m, 1:50m, and
> 1:110 million scales.  Free for use in any type of project.
>
> — https://www.naturalearthdata.com/about/terms-of-use/

The official terms of use state:
- No licence is needed to use Natural Earth data.
- Credit is appreciated but not required.

## Recommended attribution (if credited)

> Made with Natural Earth (https://www.naturalearthdata.com/)

## Use in this project

These files are used exclusively for **rendering the land mask on static
Cartopy maps** in `pages/32_Isotope_Hydrographic_Mapping.py`.  They are read
at runtime via `cartopy.io.shapereader.Reader(local_path)` — no external
Natural Earth download is performed, even in offline or air-gapped environments.

Coastline outlines (the line overlay on top of the land mask) are drawn
separately from the bundled coastline CSV files in `coastline/`, via
`envgeo_utils.plot_bundled_coastline()`.

GEBCO bathymetry data (`data_beta/GEBCO_2025_6min.nc`) is kept in a separate
directory and serves a different purpose (vertical-section depth interpolation
and seafloor estimation); it is not affected by this asset.
