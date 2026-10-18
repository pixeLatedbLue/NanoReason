# Locally served frontend assets

The dashboard makes no third-party asset requests at runtime. Files were copied
from these pinned npm release tarballs on 18 October 2026:

| Package | Version | Included assets | Notice |
| --- | --- | --- | --- |
| @fontsource-variable/geist | 5.2.8 | Latin variable WOFF2 | GEIST-LICENSE.txt (SIL Open Font License) |
| @phosphor-icons/core | 2.1.1 | Regular SVG icons | CORE-LICENSE.txt (MIT) |
| gsap | 3.13.0 | dist/gsap.min.js | Copyright and license URL retained in the library header: https://gsap.com/standard-license |

Sources are the version-specific npm registry tarballs for those packages.
The existing NanoReason brand mark is retained in index.html.
No application build or Node runtime is required to serve these assets.
manifest.json records each asset's SHA-256 digest. Update the pinned versions,
license notices and manifest together if replacing an asset.
