# High Innovation! A New Prediction Model Based on ICEEMDAN+MSCNN-BiGRU-Attention in Parallel

Preface
This issue focuses on the operation data of electric power transformer oil temperature, proposing a bidirectional gated recurrent unit (BiGRU) model based on improved ensemble empirical mode decomposition (ICEEMDAN) and multi-scale convolutional neural network (MSCNN) combined with attention mechanism. This model aims to make high-precision predictions for multi-feature time series data. The model enhances feature weighting through the GlobalAttention mechanism, improving the ability to perceive key temporal information in sequences, effectively capturing the changing patterns of oil temperature under complex operating conditions. This paper provides detailed design of the model architecture, implementation details, and experimental validation, offering significant guidance for monitoring transformer operational status and intelligent maintenance. It is also applicable to other time series prediction tasks!
1. Innovation Model Introduction
1.1 Model Evaluation:
1.2 Predictive Visualization:
1.3 Model Feature Analysis—Visualization:
We provide detailed materials, instructional documents, and video explanations, including how to replace your own dataset, tutorials on parameter adjustment, etc. The code is commented line by line, with detailed introductions to parameters:
● Data Set: Electric Power Transformer Oil Temperature Data Set
● Environment Framework: Python 3.9, PyTorch 2.1 and above are all compatible
● Model Score: Test set 0.99
● Usage Object: Paper requirements, graduation design requirements
● Code Guarantee: Code comments are detailed, ready to run without modification.
2. Innovation Points of the Model
2.1 Overall Architecture Design
In the process of modeling and predicting temperature data of electric power transformer oil, facing the complex nonlinearity, multimodality and noise interference of oil temperature signals, adopting advanced signal decomposition methods combined with multi-scale feature extraction and sequence modeling has become an effective way to improve prediction performance. This paper will introduce the key technologies used in this research: Improved Complete Ensemble Empirical Mode Decomposition (ICEEMDAN), Multi-Scale Convolutional Neural Network (MSCNN), Bidirectional Gated Recurrent Unit (BiGRU) and Global Attention based Sequence Feature Weighting method.
Input Layer: Data Preprocessing and ICEEMDAN Decomposition Module
MSCNN module branch: The Multi-Scale Convolutional Neural Network (MSCNN) utilizes parallel sets of convolutional layers, employing various kernel sizes to simultaneously extract short-term, medium-term, and long-term temporal dependencies. Subsequently, these scale features are fused to form a rich multidimensional temporal feature set.
\tBiGRU-GlobalAttention module branch: Extracts temporal bidirectional context information through a bidirectional recurrent layer, and combines global attention to focus on key time points in the sequence.
Feature Fusion Layer: Merges the feature representations of two modules to capture complementary information between the two models, enhancing the spatiotemporal feature expression capability.
Output layer: The fused features are mapped to the predicted values through a fully connected layer, achieving the prediction of oil temperature in power transformers.
This design balances long-term dependency global modeling with fine-grained capture of temporal local dynamic features, particularly suitable for time series data with significant multiscale changes, complexities, and noise.
2.2 MSCNN branch
Multi-Scale Convolutional Neural Networks (MSCNN) are a class of deep learning architectures that build upon Convolutional Neural Networks (CNN) by employing multiple convolution kernel sizes to extract features concurrently. The core idea behind MSCNN is to leverage the convolutional kernels at different scales to capture multi-level, multi-granularity information from the input data, thereby enhancing the model's expressiveness and generalization capabilities.
Multi-scale Convolutional Neural Networks (MSCNN) utilize parallel sets of convolutional layers, employing various kernel sizes (e.g., 3, 5, 7, etc.), to simultaneously extract short-term, medium-term, and long-term temporal dependencies. Subsequently, these scale features are fused to form a rich multidimensional temporal feature set. The advantages of MSCNN include:
\tCapture multi-scale time domain information: Different kernel sizes correspond to different time windows, enhancing the sensitivity to multi-level signal events.
\tEnhance model expressiveness: Multi-scale fusion reduces the defects of single-scale convolution and prevents feature omission.
\tReduce network depth pressure: Replace deep stacking with parallelism to achieve efficient feature capture.
\t2.3 BiGRU-Attention branch
Multi-feature sequence data is processed simultaneously through a BiGRU network optimized with GlobalAttention. GlobalAttention is a mechanism designed to enhance the model's focus on different parts of the input sequence. In the BiGRU model, the global attention mechanism helps the model better concentrate on the most relevant parts of the input sequence, thereby enhancing the model's performance and generalization capabilities. At each time step, the global attention mechanism calculates a weight vector indicating the model's focus on various parts of the input sequence. These weights are then applied to the feature representations output by the BiGRU, allowing for weighted consideration of all features across positions, enabling the model to more specifically focus on important temporal features. This improves the model's ability to perceive temporal features in multi-feature sequences.
3. Introduction to the Power Transformer Data Set
3.1 Import Data
3.2 ICEEMDAN Decomposition
3.3 Data Set Production and Preprocessing
Detailed explanations are provided in the documents provided!
4 Prediction Model Based on ICEEMDAN+MSCNN-BiGRU-Attention
4.1 Define the prediction network model of ICEEMDAN+MSCNN-BiGRU-Attention
4.2 Set parameters and train the model
The prediction effect of ICEEMDAN+MSCNN-BiGRU-Attention is significant, the model can fully utilize the multi-scale and multi-frequency feature extraction capabilities of MSCNN and the short-term dependency capturing ability of BiGRUGATT, with fast convergence speed, excellent performance, high prediction accuracy, and further improve the model prediction performance by appropriately adjusting model parameters.
5 Results Visualization and Model Evaluation
5.1 Prediction result visualization
5.2 Model evaluation
Previously interesting content:
Time series forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind Speed Forecast (I) Data set introduction and preprocessing
Wind Speed Forecast (II) EMD-LSTM model based on Pytorch
Wind Speed Forecast (III) EMD-LSTM-Attention model
Wind Speed Forecast (IV) EMD-Transformer model based on Pytorch
Wind Speed Forecast (V) EMD-CNN-LSTM model based on Pytorch
Wind Speed Forecast (VI) EMD-CNN-GRU parallel model based on Pytorch
CEEMDAN + Combination Prediction Model (BiLSTM-Attention + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-LSTM + ARIMA)
CEEMDAN + Combination Prediction Model (Transformer - BiLSTM + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-Transformer + ARIMA)
Multi-Feature Variable Sequence Forecasting (I) - CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (II) - CNN-LSTM-Attention Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (III) - CNN-Transformer Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (IV) Transformer-BiLSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (V) CEEMDAN+CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (VI) CEEMDAN+CNN-Transformer Wind Speed Forecasting Model
Predictive Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-BiLSTM-Attention
Predictive Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-Transformer-BiGRU
Superior Predictive Model: Secondary Decomposition-Combination Prediction
Multi-Feature Variable Sequence Forecasting (VII) CEEMDAN+Transformer-BiLSTM Forecasting Model
Multi-Feature Variable Sequence Forecasting (VIII) Predictive Model Based on Particle Swarm Optimization Algorithm - CEEMDAN-SSA-BiLSTM
Multi-Feature Variable Sequence Forecasting (11) Predictive Model Based on Pytorch TCN-GRU
VMD + CEEMDAN Secondary Decomposition, CNN-LSTM Forecasting Model
VMD + CEEMDAN Secondary Decomposition, CNN-Transformer Forecasting Model
VMD + CEEMDAN Secondary Decomposition, Transformer-BiGRU Forecasting Model
Predictive Model Based on the Optimized Algorithm SSA of Sparrows - Complete Code Kit
A powerful combination predictive model!
High Innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention Predictive Model
A powerful combination predictive model!
The lowest price online | Continuous updates for the complete kit!
Based on TCN-SENet + BiGRU-GlobalAttention Parallel Predictive Model
Secondary decomposition - Innovative predictive model family kit
BiTCN-BiGRU-CrossAttention Fusion of Spatiotemporal Features Highly Innovative Predictive Model
Combination prediction update | A powerful combination predictive model!
Parallel Predictive Model with Spatiotemporal Feature Fusion - BiTCN-Transformer
Based on Multi-level Attention Mechanism Parallel Predictive Model
 | CEEMDAN-CNN-GRU-GlobalAttention + XGBoost Combination Predictive
Multi-step prediction series | Collection of LSTM, CNN, Transformer, TCN, serial, and parallel models
END
2. Friends who like this can click to follow, give "likes" to the article, share on social circles or discussion groups

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
