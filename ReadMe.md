## **RMCR-Net**
This repo is the implementation of the paper: Recurrent Multi-View Collaborative Registration Network for 3D Reconstruction and Optical Measurement of Blade Profiles. 

Paper address:***

### Abstract
Aligning multiple viewpoints to reconstruct a complete profile is a crucial aspect of the three-dimensional optical measurement process for blade profiles. As foundational solutions, rigid pairwise point cloud registration methods often struggle to produce satisfactory results due to significant cumulative errors stemming from poor initial poses, while current multi-view registration methods also face challenges when their assumptions conflict with the practical measurement scenarios of blade profiles. To address these issues, this study proposes a learning framework to recover a transformation vector between the coordinate frames of both the optical sensor and the rotational axis in our developed optical measurement system based on extracted adaptive feature embeddings from the initial spatial representation of viewpoints, which simultaneously aligns all viewpoints into a complete profile without relying on any prerequisite assumptions. Furthermore, an elaborate recurrent updating strategy is incorporated into the framework that enables incremental refinement of the transformation vector, thus aiding the framework in meeting the high-precision measurement requirements at the 0.01mm level. Experimental results from both theoretical datasets and real-world data, conducted on three representative blades and compared against nine state-of-the-art algorithms, consistently demonstrate the superiority of the proposed method in terms of accuracy, time-efficiency, and robustness.

### Highlights
+ 1.We innovatively reformulate the 3D reconstruction of blade profiles as a quest for a transformation vector between the coordinate frames of both the optical sensor and the rotational axis in the developed optical measurement system.
+ 2.We design a learning framework to recover the transformation vector, effectively overcoming the key challenge of finding approximate point correspondences based on spatial coordinates for recovering it.
+ 3.We seamlessly integrate an elaborate recurrent updating strategy into the framework, iteratively refining the transformation vector.

### Contributions
+ 1.We reformulate the 3D reconstruction problem of blade profiles as a quest for a transformation vector between the coordinate frames of both the optical sensor and the rotational axis in our developed optical measurement system. This transformation vector aligns all viewpoints into a complete profile simultaneously, addressing the issue of error accumulation caused by sequential RPPCR from poor initial poses and eliminating the prerequisite assumptions of current multi-view registration methods. 
+ 2.We design RMCR-Net, a learning framework introduced to recover the transformation vector by extracting adaptive feature embeddings from the initial spatial representation of viewpoints. This approach effectively overcomes the challenge of finding approximate point correspondences based on spatial coordinates for recovering the transformer vector. Furthermore, we seamlessly integrate an elaborate Recurrent Updating Strategy (RUS) into the framework, iteratively refining the transformation vector and thereby mitigating biases within it.
+ 3.We assess the reliability of the proposed method through experiments conducted on three representative turbine blades. Extensive analysis and experimental results from both theoretical datasets and real-world data, compared with nine state-of-the-art algorithms, consistently demonstrate that the proposed method exhibits superior accuracy, time-efficiency, and robustness. It meets the high-precision measurement requirements at the 0.01 mm level for complex blade profiles.

### Network Architecture
![avatar](./images/网络全框架.png)

### Citation
Please cite this paper with the following bibtex:***