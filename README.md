# ASIKT
code for paper: Interpretable Knowledge Tracing with Difficulty-Aware Attention  and Selective State Space Model
## Requirements
python 3.9+  
torch 2.0.1  
numpy 1.26.0  
scikit-learn 1.5.2  
transformers  
einops  
## Data Preparation
### Datasets
The datasets can be downloaded from the links below:  
ASSIST09: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data  
ASSIST12: https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect  
ASSIST17: https://sites.google.com/view/assistmentsdatamining/dataset  
Eedi: https://eedi.com/projects/neurips-education-challenge  

### Data Format
For every 4 lines in 'dataset.csv', the format is as follows:

1. id, true_student_id
2. question_id_1, question_id_2, ...
3. skill_id_1, skill_id_2, ...
4. answer_1, answer_2, ...

Each group of four lines represents one student's interaction sequence.
Put your data in `data/your_dataset_name` using the format shown above.

### Generate Exercise Weight Matrix
Before training the model, split the dataset into training, validation, and testing sets. Then, run `data_pre.py` to generate the Exercise Weight Matrix for each set.

## Running ASIKT
An example command to train ASIKT on the ASSIST2009 dataset:
`python main.py --dataset assist2009`
