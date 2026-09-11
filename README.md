# LunarSync

LunarSync is an system that aligns Chandrayaan-2 lunar images (OHRC/TMC/IIRS) with high resolution LRO NAC reference images despite differences in scale, viewing angle, and Sun illumination.

It uses metadata guided overlap detection, scale normalization, feature matching, RANSAC verification, uniform point selection, and sub-pixel refinement to produce accurate pixel-to-pixel lunar image registration.

The project was developed for **Smart India Hackathon 2026** Problem Statement
26166: Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS).


```text
Source Product                  Reference Product
      │                                │
      ▼                                ▼
 Metadata Extraction             Metadata Extraction
      │                                │
      └──────────────┬─────────────────┘
                     ▼
              Geographic Overlap
                     │
                     ▼
             Coarse Localization
                     │
                     ▼
              Scale Normalization
                     │
                     ▼
          Sensor Specific Enhancement
                     │
                     ▼
          Coarse to Fine Matching
                     │
              ┌──────┴──────┐
              ▼             ▼
          SuperGlue        SIFT
              │             │
              └──────┬──────┘
                     ▼
             Geometric Verification
                  (RANSAC)
                     │
                     ▼
           Uniform Correspondence
                  Selection
                     │
                     ▼
             Sub Pixel Refinement
                     │
                     ▼
             Final Transformation
                     │
                     ▼
            Registered Image
                     │
                     ▼
               Evaluation
```               

## Google Colab Run:
We have implemented the modules in the form of python notebook and ran that on google colab's virtual environment with T4 GPU.
The notebook can be found in this repo named as, OHRC,_TMC_and_NAC_registration.ipynb.
We tested the notebook with OHRC <-> NAC registration. We used two datasets, ch2_ohr_ncp_20210401T2357376656_d_img_d18 for OHRC and M1350459544RE for NAC which describe the same patch. 

<table>
  <tr>
    <td align="center">
      <img src="/artifacts/ch2_ohr_ncp_20210401T2357376656_b_brw_d18.png" width="400">
      <br>
      <b>Source (OHRC)</b>
    </td>
    <td align="center">
      <img src="/artifacts/M1350459544RE.png" width="400">
      <br>
      <b>Reference (LRO NAC)</b>
    </td>
  </tr>
</table>

### BEST RUN:
On our best run we got:
| Metric                      |         Result |
| --------------------------- | -------------: |
| Raw matches                 |         **76** |
| Final inliers               |         **69** |
| Inlier ratio                |     **90.79%** |
| RMSE                        |    **0.64 px** |
| Median reprojection error   |    **0.32 px** |
| Uniform grid coverage       |     **60.94%** |

## How to set-up the Input folders:
For every source a seprate folder should be created and each patch will be stored as a folder and will be placed inside the relevent source folder. For every patch the minimum files required are,
1) .IMG
2) .png
3) .json

The IMG is the actual high resolution image of the patch.

The PNG image will be for preview(optional).

The json file must contain,
1) Product id
2) Sun elevation
3) sun_azimuth
4) Projection
5) Latitude(UL, UR, LL & LR)
6) Longitude(UL, UR, LL & LR)  