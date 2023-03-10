# Meta-Labeling:

Code base for the meta-labeling papers published with the Journal of Financial Data Science

## [Theory and Framework (Journal of Financial Data Science, Summer 2022)](https://jfds.pm-research.com/content/early/2022/06/23/jfds.2022.1.098)

**By Jacques Francois Joubert**

Meta-labeling is a machine learning (ML) layer that sits on top of a base primary strategy to help size positions, filter out false-positive signals, and improve metrics such as the Sharpe ratio and maximum drawdown. This article consolidates the knowledge of several publications into a single work, providing practitioners with a clear framework to support the application of meta-labeling to investment strategies. The relationships between binary classification metrics and strategy performance are explained, alongside answers to many frequently asked questions regarding the technique. The author also deconstructs meta-labeling into three components, using a controlled experiment to show how each component helps to improve strategy metrics and what types of features should be considered in the model specification phase.

## [Model Architectures (Journal of Financial Data Science, Fall 2022)](https://jfds.pm-research.com/content/early/2022/09/16/jfds.2022.1.108)

**By Michael Meyer, Jacques Francois Joubert, ‪Mesias Alfeus‬**

Separating the side and size of a position allows for sophisticated strategy structures to be developed. Modeling the size component can be done through a meta-labeling approach. This article establishes several heterogeneous architectures to account for key aspects of meta-labeling. They serve as a guide for practitioners in the model development process, as well as for researchers to further build on these ideas. An architecture can be developed through the lens of feature- and or strategy-driven approaches. The feature-driven approach exploits the way the information in the data is structured and how the selected models use that information, while a strategy-driven approach specifically aims to incorporate unique characteristics of the underlying trading strategy. Furthermore, the concept of inverse meta-labeling is introduced as a technique to improve the quantity and quality of the side forecasts. 

## [Calibration and Position Sizing (Journal of Financial Data Science, Spring 2023)](https://jfds.pm-research.com/content/early/2023/03/08/jfds.2023.1.119)

**By Michael Meyer, Illya Barziy, Jacques Francois Joubert**

Meta-labeling is a recently developed tool for determining the position size of a trade. It involves applying a secondary model to produce an output that can be interpreted as the estimated probability of a profitable trade, which can then be used to size positions. Before sizing the position, probability calibration can be applied to bring the model’s estimates closer to true posterior probabilities. This article investigates the use of these estimated probabilities, both uncalibrated and calibrated, in six position sizing algorithms. The algorithms used in this article include established methods used in practice and variations thereon, as well as a novel method called sigmoid optimal position sizing. The position sizing methods are evaluated and compared using strategy metrics such as the Sharpe ratio and maximum drawdown. The results indicate that the performance of fixed position sizing methods is significantly improved by calibration, whereas methods that estimate their functions from the training data do not gain any significant advantage from probability calibration.

## [Ensemble Meta-Labeling (Journal of Financial Data Science, Winter 2022)](https://jfds.pm-research.com/content/early/2022/12/14/jfds.2022.1.114)

**By Dennis Thumm, Paolo Barucca, Jacques Francois Joubert**

This study systematically investigates different ensemble methods for meta-labeling in finance and presents a framework to facilitate the selection of ensemble learning models for this purpose. Experiments were conducted on the components of information advantage and modeling for false positives to discover whether ensembles were better at extracting and detecting regimes and whether they increased model efficiency. The authors demonstrate that ensembles are especially beneficial when the underlying data consist of multiple regimes and are nonlinear in nature. The authors’ framework serves as a starting point for further research. They suggest that the use of different fusion strategies may foster model selection. Finally, the authors elaborate on how additional applications, such as position sizing, may benefit from their framework.
