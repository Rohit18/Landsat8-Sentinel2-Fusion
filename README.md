# Landsat8-Sentinel2-Fusion

**Objective:** Transform Landsat 8 spectral bands to their corresponding Sentinel-2 bands and predict the three Sentinel-2 Red Edge bands not available in Landsat 8. Additionally, increase the availability of Sentinel-2 scenes potentially by 30% by fusing the dataset with Landsat 8.

**Issue:** Data availability can be an issue due to the relatively lower temporal resolution and cloud cover.

**Previous Work and Limitations:** Previous work on fusing Landsat 8 and Sentinel-2 only works with the common spectral bands between L8 and S2 and does not provide a solution to predict the additional Sentinel-2 spectral bands such as Red Edge 1, 2, and 3 which help in the extraction of certain phenological properties.

**Possible Solution:** Generative Adversarial Networks are known to learn the data distribution of the target dataset (Sentinel-2) in a supervised manner and transform the samples from the input dataset (Landsat 8) to replicate the corresponding sample from the target dataset (Sentinel-2). We will train a GAN to learn the data distribution of the Red Edge bands from the Landsat 8 bands informationally closest to the Sentinel-2 Red Edge bands (Green for Red Edge 1 and NIR for Red Edge 2 and 3).

**L2SGAN** or Landsat 8 to Sentinel-2 Generative Adversarial Network will be compared with a deep residual encoder decoder architecture **DREDN** to highlight the pros and cons of using a GAN over other previously used architectures for satellite image tasks.

![Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Methodology.png](https://github.com/Rohit18/Landsat8-Sentinel2-Fusion/blob/main/Images/Methodology.png)

**Results:**

**Landsat 8 Green to Sentinel-2 Green**

A: Landsat 8 Green, B: Sentinel-2 like Green by GAN, C: Sentinel-2 like Green by DREDN, D: Original Sentinel-2 Green

![Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Result.png](https://github.com/Rohit18/Landsat8-Sentinel2-Fusion/blob/main/Images/Result.png)

**Landsat 8 NIR to Sentinel-2 Red Edge 1**

A: Landsat 8 NIR, B: Sentinel2 like NIR by GAN, C: Sentinel2 like NIR by DREDN, D: Original Sentinel-2 NIR

![Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Result2.png](https://github.com/Rohit18/Landsat8-Sentinel2-Fusion/blob/main/Images/Result2.png)


|     S2 G     |     ERGAS      |     SAM       |     SCC       |     PSNR     |     RMSE     |     UQI       |
|--------------|----------------|---------------|---------------|--------------|--------------|---------------|
|     L8 G     |     2330.51    |     0.2376    |     0.0632    |     22.86    |     21.05    |     0.9351    |
|     DREDN    |     1931.55    |     0.2034    |     0.1898    |     24.95    |     16.99    |     0.9525    |
|     GAN      |     1870.25    |     0.2052    |     0.1829    |     24.86    |     17.15    |     0.9526    |


|     S2 RE1    |     ERGAS      |     SAM       |     SCC       |     PSNR     |     RMSE     |     UQI       |
|---------------|----------------|---------------|---------------|--------------|--------------|---------------|
|     L8 G      |     3597.13    |     0.2211    |     0.0631    |     20.98    |     24.71    |     0.8650    |
|     DREDN     |     1712.56    |     0.1725    |     0.1580    |     23.60    |     18.60    |     0.9393    |
|     GAN       |     1660.75    |     0.1677    |     0.1582    |     24.07    |     17.35    |     0.9484    |


|     S2 NIR    |     ERGAS     |     SAM       |     SCC       |     PSNR     |     RMSE     |     UQI       |
|---------------|---------------|---------------|---------------|--------------|--------------|---------------|
|     L8 NIR    |     918.57    |     0.1279    |     0.2588    |     24.39    |     16.40    |     0.9809    |
|     DREDN     |     780.14    |     0.1106    |     0.3970    |     26.05    |     13.47    |     0.9869    |
|     GAN       |     848.66    |     0.1227    |     0.3238    |     25.37    |     14.88    |     0.9853    |

|     S2 RE2    |     ERGAS      |     SAM       |     SCC       |     PSNR     |     RMSE     |     UQI       |
|---------------|----------------|---------------|---------------|--------------|--------------|---------------|
|     L8 NIR    |     1399.44    |     0.1865    |     0.2276    |     20.74    |     25.53    |     0.9480    |
|     DREDN     |     1148.09    |     0.1670    |     0.3454    |     23.27    |     19.10    |     0.9678    |
|     GAN       |     1176.92    |     0.1751    |     0.3034    |     22.98    |     19.84    |     0.9670    |

|     S2 RE3    |     ERGAS      |     SAM       |     SCC       |     PSNR     |     RMSE     |     UQI       |
|---------------|----------------|---------------|---------------|--------------|--------------|---------------|
|     L8 NIR    |     1096.80    |     0.1426    |     0.2480    |     22.58    |     19.94    |     0.9716    |
|     DREDN     |     869.04     |     0.1228    |     0.3850    |     25.25    |     14.79    |     0.9838    |
|     GAN       |     1081.82    |     0.1454    |     0.2946    |     23.40    |     18.12    |     0.9760    |

