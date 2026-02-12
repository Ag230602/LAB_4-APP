# Doc 1

This is a placeholder document. Replace with your Lab-3 dataset content.

Key facts:
- Fact A
- Fact B
- # Doc 1

This is a placeholder document. Replace with your Lab-3 dataset content.

Key facts:
- Fact A
- Fact B
GenAI-Driven Uncertainty-Aware Forecasting and Recovery
Simulation for Decision Support
Adrija Ghosh (GitHub: @Ag230602)
University of Missouri–Kansas City
Abstract
High-stakes decision-making in disaster and emergency management requires forecasts that are not only
accurate but also interpretable under uncertainty. Traditional forecasting systems provide probabilistic outputs
that are often difficult for human decision-makers to translate into action. This project proposes a GenAI-driven,
uncertainty-aware forecasting system that integrates probabilistic spatio-temporal models with
Retrieval-Augmented Generation (RAG) to produce scenario-based explanations. Built on a Snowflake-centered
analytics architecture, the system ingests environmental data, quantifies predictive uncertainty, and generates
human-readable narratives that support early and confident decision-making. The final deliverable is an
interactive prototype demonstrating forecast outputs, uncertainty explanations, and recovery-oriented decision
support.
Index Terms
Uncertainty Quantification, Generative AI, Spatio-Temporal Forecasting, Decision Support Systems,
Retrieval-Augmented Generation, Snowflake
I. Introduction
Emergency planners and disaster response agencies must often act on forecasts that are inherently uncertain.
While modern machine learning models can generate accurate predictions, they frequently fail to communicate
uncertainty in a way that is actionable for human users. This gap leads to delayed responses, misallocation of
resources, and reduced trust in automated systems. Recent advances in probabilistic forecasting and large
language models create an opportunity to bridge this gap by translating uncertainty into structured,
scenario-based explanations.
II. Project Objectives and Scope
The objective of this project is to design and implement an uncertainty-aware forecasting system that improves
early decision-making in disaster and emergency planning. Target users include emergency managers,
infrastructure planners, and policy analysts. The innovation lies in combining probabilistic forecasting with
GenAI-based explanation, allowing uncertainty to be communicated through grounded narratives rather than
abstract statistics.
III. System Architecture
The system follows a modular architecture centered on Snowflake. Spatio-temporal environmental data are
ingested through Snowflake stages and Snowpipe, processed using Snowpark for forecasting and uncertainty
quantification, and stored alongside embeddings for retrieval. A Retrieval-Augmented Generation pipeline uses
large language models to generate scenario explanations that are grounded in forecast outputs and historical
evidence.
IV. Related Work (NeurIPS 2025)
This project builds upon recent research in probabilistic forecasting, uncertainty quantification, and
retrieval-augmented generation. Key influences include influence-guided context selection for RAG, cooperative
RAG frameworks for multi-step reasoning, and scalable methods for constructing adaptive prediction bands.
These works inform both the forecasting backbone and the explanation layer of the proposed system.
V. Data Sources
Three categories of datasets are used: (1) spatio-temporal environmental forecasting data from Kaggle, (2)
climate risk and vulnerability indicators from Hugging Face Datasets, and (3) historical disaster event records
from government open-data APIs. Data are ingested into Snowflake using external stages and Snowpipe, with
batch and incremental ingestion strategies.
VI. Methods, Technologies, and Tools
Forecasting models include ensemble and probabilistic approaches that produce prediction intervals and
uncertainty estimates. Snowpark is used for scalable data processing and model execution. The GenAI layer
implements Retrieval-Augmented Generation, combining vector embeddings with large language models to
produce grounded scenario explanations. Optional rule-based validation layers ensure consistency between
numeric forecasts and generated narratives.
VII. Infrastructure and Deployment
The system is deployed as a cloud-native application with a Streamlit-based dashboard. Snowflake serves as the
data and analytics backbone, while external LLM APIs or Snowflake Cortex provide generative capabilities.
Monitoring pipelines track forecast accuracy, uncertainty drift, latency, and explanation faithfulness.
VIII. Expected Outcomes and Evaluation
The live demo will present an interactive dashboard displaying spatio-temporal forecasts, uncertainty levels, and
scenario-based explanations. Evaluation metrics include forecast accuracy, calibration, scalability, latency,
explainability, and user trust.
IX. Reproducibility and Deliverables
All data schemas, Snowflake SQL scripts, Snowpark notebooks, and evaluation pipelines are documented in the
project GitHub repository. The final deliverables include a working prototype, full documentation, and an
evaluation report, ensuring end-to-end reproducibility.
X. Conclusion
This project advances forecasting systems from predictive tools to decision-centric intelligence platforms. By
integrating uncertainty-aware modeling with GenAI-driven explanation, the system supports earlier, more
confident, and more transparent decision-making in disaster and emergency planning contexts.

