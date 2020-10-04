# Landsat8-Sentinel2-Fusion

**Objective:** Transform Landsat 8 spectral bands to their corresponding Sentinel-2 bands and predict the three Sentinel-2 Red Edge bands not available in Landsat 8. Additionally, increase the availability of Sentinel-2 scenes potentially by 30% by fusing the dataset with Landsat 8.

**Issue:** Data availability can be an issue due to the relatively lower temporal resolution and cloud cover.

**Previous Work and Limitations:** Previous work on fusing Landsat 8 and Sentinel-2 only works with the common spectral bands between L8 and S2 and does not provide a solution to predict the additional Sentinel-2 spectral bands such as Red Edge 1, 2, and 3 which help in the extraction of certain phenological properties.

**Possible Solution:** Generative Adversarial Networks are known to learn the data distribution of the target dataset (Sentinel-2) in a supervised manner and transform the samples from the input dataset (Landsat 8) to replicate the corresponding sample from the target dataset (Sentinel-2). We will train a GAN to learn the data distribution of the Red Edge bands from the Landsat 8 bands informationally closest to the Sentinel-2 Red Edge bands (Green for Red Edge 1 and NIR for Red Edge 2 and 3).

**L2SGAN** or Landsat 8 to Sentinel-2 Generative Adversarial Network will be compared with a deep residual encoder decoder architecture **DREDN** to highlight the pros and cons of using a GAN over other previously used architectures for satellite image tasks.

![Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Methodology.png](Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Methodology.png)

**Results:**

**Landsat 8 Green to Sentinel-2 Green**

A: Landsat 8 Green, B: Sentinel-2 like Green by GAN, C: Sentinel-2 like Green by DREDN, D: Original Sentinel-2 Green

![Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Result.png](Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Result.png)

**Landsat 8 NIR to Sentinel-2 Red Edge 1**

A: Landsat 8 NIR, B: Sentinel2 like NIR by GAN, C: Sentinel2 like NIR by DREDN, D: Original Sentinel-2 NIR

![Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Result2.png](Landsat8-Sentinel2-Fusion%2065454290927549219c061f53212d6fd8/Result2.png)

[Untitled](https://www.notion.so/bd775ed5e211460ba7262434e79d47ee)

[Untitled](https://www.notion.so/7331357d753a4683b13397acb290e055)

[Untitled](https://www.notion.so/8b15a23d7647431c9dccd4b0e9bd5095)

[Untitled](https://www.notion.so/c2a91a1e0c0c482fb23885ae95eb2efe)

[Untitled](https://www.notion.so/4d1f8610de6c4177a1c42e9a6e0a6cd9)