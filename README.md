# QUANTUM IQ HACKATHON 🚀
### Prajna AI - Wizzify Your Data  
**© Prajna AI - Confidential & Proprietary**
FRONTEND REPO : https://github.com/Asachdeva001/ieee-hacky-front

Welcome to **Quantum IQ Hackathon**, a challenge designed to empower participants in building a next-generation PDF ingestion and querying system. This system should allow users to extract insights from PDFs with ease, automatically suggest relevant questions, answer user queries, and provide precise citations for validation.

## 🚩 **Problem Overview**
In today's world, where digital information is abundant, the need for managing and querying vast amounts of PDF documents is critical for businesses and researchers. However, extracting meaningful insights and validating facts from unstructured data contained in PDF files remains a challenge.

The **Quantum IQ Hackathon** challenge involves building a comprehensive system to:
1. Upload and parse PDFs.
2. Generate semantic embeddings for better querying.
3. Suggest relevant questions based on content.
4. Answer user queries with citations for validation.
5. Provide a seamless user interface, deployable in a cloud environment.

## 🎯 **Objective**
The goal is to develop an **Intelligent PDF Querying System (IPQS)** with the following capabilities:
- **PDF Document Ingestion:** Seamlessly upload and parse PDFs.
- **Embedding Generation:** Create semantic embeddings and store them for querying.
- **Question Suggestion Engine:** Automatically generate insightful questions based on the uploaded PDFs.
- **User Query Interface:** Allow users to submit natural language queries and receive accurate answers.
- **Citation and Validation:** Provide precise citations for every query response.
- **Interactive Frontend:** Design a user-friendly web interface for interaction.
- **Cloud Deployment:** Ensure the system can be deployed in a cloud environment for public access.

## 💡 **Key Features**
1. **PDF Document Ingestion**
    - Users can upload PDF files, which are processed to extract content and metadata.
    - **Evaluation:** Can the system handle complex PDF structures while preserving important information?

2. **Embedding Generation and Data Persistence**
    - Generate embeddings that capture the semantic meaning of the PDF content and store them for efficient querying.
    - **Evaluation:** How effectively can the system retrieve and query relevant information based on embeddings?

3. **Question Suggestion Engine**
    - Automatically suggest 3-5 insightful questions related to the uploaded PDF content.
    - **Evaluation:** The quality, relevance, and diversity of the generated questions.

4. **User Querying Interface**
    - Users can submit natural language queries related to the PDF content and receive concise answers.
    - **Evaluation:** How accurate and relevant are the responses to user queries?

5. **Citation and Validation**
    - Provide a citation for every response, including specific page or section references within the PDF.
    - **Evaluation:** Can the system accurately trace answers back to the original source in the PDF?

6. **Interactive Frontend**
    - An intuitive web interface that supports document uploads, question suggestions, and query submissions.
    - **Evaluation:** Usability, responsiveness, and design appeal of the interface.

7. **Deployment**
    - Deploy the entire system on a cloud platform (e.g., AWS, Vercel) ensuring scalability and accessibility.
    - **Evaluation:** Functionality of the deployment, ease of access, and performance under user load.

## 🌟 **Additional Features (Bonus)**
- **Cost-Effective API Usage:** Optimize API usage for minimal cost without sacrificing accuracy.
- **Support for Additional Document Types:** Extend support to formats like HTML, CSV, and Excel.
- **Customizable User Dashboard:** Provide users with a dashboard to manage their documents, query history, and FAQs.

## 🔧 **System Requirements**
- **PDF Parsing**: Efficient extraction and structure preservation of content.
- **Embedding Engine**: Implement semantic embedding models (e.g., ChromaDB) for querying.
- **Frontend**: Built with a responsive design to ensure seamless user interaction.
- **Backend**: Capable of handling user queries and providing real-time responses.
- **Deployment**: Deployed on a cloud platform with minimal downtime and fast response times.

## 📦 **Technologies Used**
- **Frontend**: Streamlit (for rapid prototyping and user interaction)
- **Backend**: Langchain, FAISS (for handling queries and storing embeddings)
- **Model**: Google Generative AI for question answering and embedding generation
- **Cloud**: Vercel or AWS (for deployment)

## 💻 **Getting Started**

### Prerequisites
- Python 3.x
- `pip` for package management
- API keys (Google Generative AI, etc.)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/quantum-iq-hackathon.git
    cd quantum-iq-hackathon
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables for API keys:
    ```bash
    export GOOGLE_API_KEY=your-google-api-key
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

### Deployment
To deploy the app on a cloud platform like **Vercel** or **AWS**, follow these steps:
1. Configure your cloud service (Vercel, AWS).
2. Deploy the app using the platform's CLI or web interface.
3. Ensure proper environment variables are set for API usage in the cloud.

## 📄 **Usage**
1. **Upload PDF Documents**: Start by uploading one or more PDFs.
2. **Suggested Questions**: See automatically generated questions related to the content of the PDFs.
3. **Ask Your Queries**: Submit natural language queries to extract specific information.
4. **Citations**: Validate the results by checking citations for the source within the PDF.

## 🌐 **Demo**
**[Live Demo Link](#)** - Coming soon!

## 🤝 **Contributing**
We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## 🛠 **License**
This project is licensed under **© Prajna AI - Confidential & Proprietary**.
