# Meta-Labeling:

Code base for the meta-labeling papers published with the Journal of Financial Data Science

## [Theory and Framework (Journal of Financial Data Science, Summer 2022)](https://jfds.pm-research.com/content/early/2022/06/23/jfds.2022.1.098)

**By Jacques Francois Joubert**

Meta-labeling is a machine learning (ML) layer that sits on top of a base primary strategy to help size positions, filter out false-positive signals, and improve metrics such as the Sharpe ratio and maximum drawdown. This article consolidates the knowledge of several publications into a single work, providing practitioners with a clear framework to support the application of meta-labeling to investment strategies. The relationships between binary classification metrics and strategy performance are explained, alongside answers to many frequently asked questions regarding the technique. The author also deconstructs meta-labeling into three components, using a controlled experiment to show how each component helps to improve strategy metrics and what types of features should be considered in the model specification phase.

## [Model Architectures (Journal of Financial Data Science, Fall 2022)](https://jfds.pm-research.com/content/early/2022/09/16/jfds.2022.1.108)

**By Michael Meyer, Jacques Francois Joubert, ‪Mesias Alfeus‬**

Separating the side and size of a position allows for sophisticated strategy structures to be developed. Modeling the size component can be done through a meta-labeling approach. This article establishes several heterogeneous architectures to account for key aspects of meta-labeling. They serve as a guide for practitioners in the model development process, as well as for researchers to further build on these ideas. An architecture can be developed through the lens of feature- and or strategy-driven approaches. The feature-driven approach exploits the way the information in the data is structured and how the selected models use that information, while a strategy-driven approach specifically aims to incorporate unique characteristics of the underlying trading strategy. Furthermore, the concept of inverse meta-labeling is introduced as a technique to improve the quantity and quality of the side forecasts. 

## Calibration and Position Sizing (Working Paper)

**By Illya Barziy, Jacques Francois Joubert, Michael Meyer**

Meta-labeling allows for a new mechanism to determine the position size of a trade. The secondary model produces an estimate of the probability of a profitable trade, that can be used to size the positions. Probability calibration can also be applied to transform these estimates closer to true posterior probabilities as an in-between step, before sizing the position. This article investigates six position sizing algorithms from predicted probabilities for uncalibrated and calibrated probabilities. The algorithms used in this article are established methods used in practice and variations thereof, and one novel method is proposed called Sigmoidal Optimal Position Sizing (SOPS). The position sizing methods are evaluated and compared through strategy metrics such as the Sharpe ratio and maximum drawdown. The results indicate that the various methods all have unique advantages and drawbacks. Furthermore, it is shown that fixed position sizing methods significantly gain performance when calibration is applied, while methods that estimate their functions from the training data do not gain significant advantage from applying probability calibration.

## Ensemble Model Selection Framework for Meta-Labeling (Working Paper)

**By Dennis Thumm, Jacques Francois Joubert, Paolo Barucca**

This dissertation investigates the development of a framework to select machine learning models for meta-labeling and how ensemble learning can be incorporated. Meta-labeling consists of a primary model generating classifications and a secondary model labelling those classifications in terms of their correctness. For the meta-labeling ensembles, different models are selected and compared in the experiments. Since how to select those models is the research subject, the aim is to provide a guide to combining secondary models with ensemble learning.
