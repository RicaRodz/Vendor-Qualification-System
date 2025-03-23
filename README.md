# **Vendor Qualification System**  


## **Overview**  
The **Vendor Qualification System** is a lightweight yet powerful tool designed to evaluate software vendors based on feature similarity. By processing a CSV file, the system ranks vendors using advanced embedding-based similarity calculations, offering a more nuanced understanding of feature relevance.  

## **Approach**  
To achieve accurate vendor qualification, I leveraged **embeddings** for semantic reasoning. This method allows for a deeper contextual comparison between user-input capabilities and vendor-provided features, leading to more meaningful rankings.  

## **Challenges Faced**  
Extracting nested features from the dataset required additional preprocessing.  
Choosing between **embeddings** and **TF-IDF** was a critical decision—ultimately, embeddings were chosen for a richer feature representation.  


## **Future Improvements**  
**API Development** – Transforming the system into an accessible API for seamless integration.  
**Enhanced Similarity Search** – Refining the ranking algorithm for even more accurate results.  
**Vector Database Integration** – Leveraging a vector database for faster lookups, improving efficiency and scalability.  
* I would also containarize the application 

---
### ** Note:**  
This project originally started in a Jupyter notebook, where I carefully documented my thought process, experiments, and step-by-step progress in building the system. That notebook serves as a detailed record of the development journey. To ensure ease of use and a more structured application, the final, refined system has been organized inside the `app` folder, ready to run seamlessly. 🚀

## How to Run This Project  

### **1️⃣ Clone the Repository**  

### **2️⃣ Place Data Files in the `data/` Folder**  
Ensure that the **pickle file** (`vendors_data_with_embeddings.pkl`) and G2 software product overview.csv are inside the `data` folder at the root of the project:  

```
vendor-qualification-system/
│── app/
│── data/
│   ├── vendors_data_with_embeddings.pkl  # Required Data File
│   ├── G2 software product overview.csv # Required for jupyterNotebook
│── test/
│── venv/
│── requirements.txt
│── README.md
```

---

#### **1. Create a Virtual Environment**  
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **2. Install Dependencies**  
```sh
pip install -r requirements.txt
```

#### **3. Run the Application**  
```sh
python app/main.py
```

---
