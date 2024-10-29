import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import os
from dotenv import load_dotenv
from datetime import datetime
import time
load_dotenv()

# from llm_lib import log_timestamp
# from supabase_lib import gemini_to_supabase
# from pgvector_lib import insert_image_to_pg
# from log import log_timestamp

def read_image(image_path):
  with open(image_path, "rb") as f:
    image_data = f.read()
  return image_data

# Given the task of analyzing the image. The image is of SAP Fiori application dedicated to SWIFT Project. 

prompt_baseline = """

Given the task of analyzing and validating the output of an UI Automator agent. The image is of Screenshot of a Desktop.
Analyze the screen Extract the information in the expected output format.
Responsd back in JSON format as given below so that the data can be saved in data.json file.

Expected Output - 
{
    'application_name': '',
    
    'application_details': '',
    
    'ui_filters': '',
    
    'input_fields': [ 
        {
            'label': '',
            'value': '',
            'mandatory': '', # yes/no
            'business_relavance': ''
        }
    ],
    
    'section_headers': [
        {
            'label': '',
            'business_relavance': ''
        }
    ],
    
    'chart': [
        {
            'label': '',
            'summerization': '',
            'business_relavance': ''
        }        
    ],
    
    'table': {
      'table_name': '',
      'table_description': '',
      'table_data': [
          {
              'column1': 'value1',
              'column2': 'value2'
          },
          {
              'column1': 'value3',
              'column2': 'value4'
          }          
      ]  
    },
    
    'ui_errors': [
        {
            'description': '',
            'reason': ''
        }
    ]
}
---------

Instructions - 
 - application_name: name of Fiori application. example - Invoice Approval, Workflow Monitor, Create Engagement etc.
 - application_details: functionality of this application and business relavance 
 - ui_filters: if user selected some filters for fetching the data
 - fields: this is the list of fields inside the screen
 - section_headers: sections in the Fiori application. example - Workflow KPI
 - chart: if any chart visible then fill this details
 - table: fill the details of the table with the description if visible
 - ui_errors: if any errors are visible then fill this details
"""

# Function to convert image to base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_encoded

def generate_sc_desc(image_path,prompt):
    if prompt != '':
        try: 
            # log_timestamp('GEMINI-START: ')
            #   image_path = filename
            image_data = read_image(image_path)
            image1 = Part.from_data( mime_type="image/png", data=image_data)
            
            generation_config = {
                "max_output_tokens": 8192,
                "temperature": 1,
                "top_p": 0.95,
            }

            safety_settings = {
                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            vertexai.init(project="genai-sabarna", location="us-central1")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="genai-sabarna-cb0c7b7a01ef.json"
            
            model = GenerativeModel("gemini-1.5-flash-preview-0514")
            
            responses = model.generate_content(
                [image1,prompt],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )
            
            data = ''
            for response in responses:
                data = data + response.text
                # print(response.text, end="")  
            #   print(data)
            # log_timestamp('GEMINI-END: ')
            return data
        except Exception as err:
            print(err)
            return ''
    else:
        return ''

def process_image_through_gemini(user_id,file_location):
    # Save the uploaded image to a file or process it

    gemini_data = ''
    skip_gemini = ''
    try:
        skip_gemini = os.environ['SKIP_GEMINI']
    except Exception as err:
        skip_gemini = ''
    filename = os.path.basename(file_location)
    dt = datetime.fromtimestamp(time.time())
    formatted_timestamp = dt.strftime("%Y%m%d%H%M%S")
    # print('timestamp: ',formatted_timestamp)
    image_base64 = convert_image_to_base64(file_location)
    json_data = {
        "filename": filename,
        "timestamp": formatted_timestamp,
        "image_data": image_base64
    }
    
    # if(skip_gemini == 'N'):
    # log_timestamp('IMAGE-PUSH-START: ')
    # gemini_to_supabase(user_id,gemini_data,json_data)
    # insert_image_to_pg(user_id,gemini_data,json_data)
    # log_timestamp('IMAGE-PUSH-STOP: ')
    return True


# def process_image_through_gemini(user_id,file_location):
#     # Save the uploaded image to a file or process it

#     gemini_data = ''
#     skip_gemini = ''
#     # file_location_txt = file_location + '.txt'
#     try:
#         skip_gemini = os.environ['SKIP_GEMINI']
#     except Exception as err:
#         skip_gemini = ''


# # Get the filename from the path
#     filename = os.path.basename(file_location)

# # Get the current timestamp in ISO format
#     # timestamp = time.time()
#     dt = datetime.fromtimestamp(time.time())
#     formatted_timestamp = dt.strftime("%Y%m%d%H%M%S")
#     print('timestamp: ',formatted_timestamp)

# # Convert image to base64
#     image_base64 = convert_image_to_base64(file_location)

# # Create JSON object with additional fields
#     json_data = {
#         "filename": filename,
#         "timestamp": formatted_timestamp,
#         "image_data": image_base64
#     }
    
            
#     if(skip_gemini == 'N'):
#         # gemini_data = generate_sc_desc(file_location,prompt_baseline)    
#         gemini_to_supabase(user_id,gemini_data,json_data)