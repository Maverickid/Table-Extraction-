import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, TableTransformerForObjectDetection
from paddleocr import PaddleOCR
from shapely.geometry import Polygon, Point
import sqlite3
import re
from pyzbar.pyzbar import decode
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration,  WebRtcMode
import av


# Connect to the SQLite3 database
conn = sqlite3.connect('database.db')

# Function to save DataFrame to SQLite3 database
def save_dataframe_to_sqlite(fd, table_name, overwrite=False):
    if fd is not None:
        fd = clean_dataframe(fd)  # Clean the DataFrame before saving
        fd.reset_index(drop=True, inplace=True)  # Reset the index
        cursor = conn.cursor()
        
        # Sanitize table name
        table_name = ''.join(char for char in table_name if char.isalnum() or char == '_')
        if not table_name:
            st.error("Table name is invalid after sanitization. Please use only letters, numbers, and underscores.")
            return

        # Check if table name starts with a digit
        if table_name[0].isdigit():
            st.error("Table name cannot start with a digit. Please choose a different name.")
            return
        
        # If overwrite is True and table exists, drop the existing table
        if overwrite and table_exists(table_name):
            cursor.execute(f"DROP TABLE {table_name}")
            print('Table dropped')
        
        # Create table if it doesn't exist
        columns = ', '.join([f"{col.replace(' ', '_').replace('-', '_')} TEXT" for col in fd.columns])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
        print('Table created')
        
        # Convert DataFrame to SQL insert statements
        for idx, row in fd.iterrows():
            placeholders = ', '.join(['?' for _ in row])
            cursor.execute(f"INSERT INTO {table_name} ({', '.join([col.replace(' ', '_').replace('-', '_') for col in fd.columns])}) VALUES ({placeholders})", tuple(row))
        
        
        # Commit the transaction
        conn.commit()
        cursor.close()
        st.success(f"DataFrame saved as '{table_name}'")
    else:
        st.error("Edit not completed. Action is redundant")

# Function to check if a table exists in the database
def table_exists(table_name):
    cursor = conn.cursor()
    
    # Sanitize table name
    table_name = ''.join(char for char in table_name if char.isalnum() or char == '_')
    if not table_name:
        return False
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cursor.fetchone() is not None
    cursor.close()
    return exists

# Function to retrieve DataFrame from SQLite3 database
def retrieve_dataframe_from_sqlite(table_name):
    cursor = conn.cursor()
    
    # Retrieve data from table
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    
    cursor.close()
    return df


# Function to apply Excel-like formula to DataFrame
def apply_formula_to_dataframe(df, formula):
    
    formula_columns = set(re.findall(r'([a-zA-Z_]+)', formula))  # Assuming column names only contain alphabets and underscore
    
    # Convert columns mentioned in the formula to numeric types where possible
    for col in formula_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    try:
        # Parse the formula to extract the new column name and the expression
        new_column, expression = formula.split('=', 1)
        new_column = new_column.strip()
        expression = expression.strip()
        
        # Check if the expression is an IF statement
        if expression.lower().startswith('if'):
            # Parse the formula to extract the condition and the true/false expressions
            condition, true_expr, false_expr = parse_if_formula(expression)
            # Apply the condition using np.where
            result = np.where(eval(condition, {'np': np, **df.to_dict(orient='series')}), eval(true_expr, {'np': np, **df.to_dict(orient='series')}), eval(false_expr, {'np': np, **df.to_dict(orient='series')}))
            df[new_column] = result
        else:
            # Use eval for non-IF formulas
            df.eval(f'{new_column} = {expression}', inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error occurred while applying formula: {str(e)}")
        return None

# Function to parse IF formulas
def parse_if_formula(formula):
    # Strip 'IF(' and the outermost ')'
    formula = formula[3:-1].strip()
    
    # Find the first comma outside nested IFs
    stack = 0
    for i, char in enumerate(formula):
        if char == '(':
            stack += 1
        elif char == ')':
            stack -= 1
        elif char == ',' and stack == 0:
            break
    
    condition = formula[:i].strip()
    remaining = formula[i+1:].strip()

    # Find the next comma separating true and false expressions
    stack = 0
    for j, char in enumerate(remaining):
        if char == '(':
            stack += 1
        elif char == ')':
            stack -= 1
        elif char == ',' and stack == 0:
            break
    
    true_expr = remaining[:j].strip()
    false_expr = remaining[j+1:].strip()
    
    # Check if false_expr is another IF statement
    if false_expr.lower().startswith('if'):
        false_expr = parse_if_formula(f'IF({false_expr})')
    
    return condition, true_expr, false_expr

def convert_dataframe_dtypes(df):
    new_df = df.copy()
    return new_df

def clean_dataframe(df):
    # Ensure column names are valid SQL identifiers
    df.columns = [re.sub(r'\W|^(?=\d)', '_', col) for col in df.columns]

    # Ensure there are no duplicate column names
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected after sanitization.")

    # Handle missing values by replacing them with None (which will be treated as NULL in SQL)
    df = df.where(pd.notnull(df), None)

    # Ensure data types are compatible with SQLite
    df = df.astype(str)

    return df

# Function to delete a table from the database
def delete_table_from_sqlite(table_name):
    cursor = conn.cursor()
    
    # Sanitize table name
    table_name = ''.join(char for char in table_name if char.isalnum() or char == '_')
    if not table_name:
        st.error("Table name is invalid after sanitization. Please use only letters, numbers, and underscores.")
        return
    
    if table_exists(table_name):
        cursor.execute(f"DROP TABLE {table_name}")
        conn.commit()
        st.success(f"Table '{table_name}' deleted successfully.")
    else:
        st.error(f"Table '{table_name}' does not exist.")
    
    cursor.close()

# Initialize PaddleOCR
# ocr = PaddleOCR(use_gpu=False, use_angle_cls=True, lang="en")
@st.cache_resource
def load_ppocr_model():
    # Load the OCR model
    ocr = PaddleOCR(
        use_gpu=False,
        use_angle_cls=True,
        lang='en'
    )
    return ocr

# Function to load the Table Transformer Detection model
@st.cache_resource
def load_table_transformer_detection_model():
    try:    
        model_name = "microsoft/table-transformer-detection"
        model = DetrForObjectDetection.from_pretrained(model_name)
        feature_extractor = DetrImageProcessor.from_pretrained(model_name)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Failed to load Table Transformer model: {e}")
        return None, None

@st.cache_resource
def load_table_transformer_structure_model():
    try:
        model_name = "microsoft/table-transformer-structure-recognition"
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
        feature_extractor = DetrImageProcessor.from_pretrained(model_name)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Failed to load Table Transformer model: {e}")
        return None, None


def process_image_with_model(image, feature_extractor, model):
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    return outputs

# Function to crop the table from the image
# def crop_table_from_image(image, outputs, feature_extractor,model):
#     target_sizes = [image.size[::-1]]
#     results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
#     label_dict = model.config.id2label
#     table_boxes = []

#     for label, box in zip(results['labels'], results['boxes']):
#         label_name = label_dict[label.item()]
#         if label_name == "table":
#             table_boxes.append(box.tolist())

#     if table_boxes:
#         # Assuming the first detected table is the one we want to crop
#         box = table_boxes[0]
#         x_min, y_min, x_max, y_max = map(int, box)
#         cropped_image = image.crop((x_min, y_min, x_max, y_max))
#         return cropped_image
#     else:
#         return image  # If no table is detected, return the original image

# Function to extract text within a polygon
def extract_text_within_polygon(results, polygon, pad1=0, pad2=0, target_size=(200, 200)):
    extracted_texts = []
    for result_item in results:
        for result in result_item:
            bbox = result[0]  # Bounding box coordinates
            text = result[1][0]  # Extracted text
            bbox_center = Point((bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[2][1]) / 2)
            if polygon.contains(bbox_center):
                extracted_texts.append(text)
    extracted_text = ' '.join(str(text) for text in extracted_texts)
    return extracted_text

# Function to extract columns and display Excel
def extract_columns_and_display_excel(results, image, row_boxes, column_boxes):
    # Initialize a dictionary to store the extracted text for each column
    column_texts = {}

    # Iterate over row boxes sorted by top coordinate
    for row_box in sorted(row_boxes, key=lambda x: x[1]):
        row_polygon = Polygon([(row_box[0], row_box[1]), (row_box[2], row_box[1]),
                               (row_box[2], row_box[3]), (row_box[0], row_box[3])])

        # Iterate over column boxes sorted by their x-coordinate
        for i, column_box in enumerate(sorted(column_boxes, key=lambda x: x[0])):
            column_polygon = Polygon([(column_box[0], column_box[1]), (column_box[2], column_box[1]),
                                      (column_box[2], column_box[3]), (column_box[0], column_box[3])])
            column_name = f"column_{i + 1}"

            # Initialize column_texts dictionary if key doesn't exist
            if column_name not in column_texts:
                column_texts[column_name] = []

            # Check if column box intersects with row box
            if column_polygon.intersects(row_polygon):
                intersection = column_polygon.intersection(row_polygon)

                # Extract text within the intersection polygon
                text_info = extract_text_within_polygon(results, intersection, -1, +1)

                # Save the extracted text
                column_texts[column_name].append(text_info)
    
    # Display Excel
    column_data = {}
    for column_name, text_list in column_texts.items():
        if not text_list:  # Skip empty columns
            continue
        if text_list[0] == "":
            column_header = "Empty header"
        else:
            column_header = text_list[0]
        column_values = text_list[1:]
        column_data[column_header] = column_values
    
    df = pd.DataFrame(column_data)
    df_correctedfordtypes = convert_dataframe_dtypes(df)
    # st.dataframe(df_correctedfordtypes)  # Display DataFrame
    return df_correctedfordtypes

# Function to check databases for corresponding column value
def check_databases(df, scanned_data, check_column="Any", display_column=None):
    if check_column == "Any":
        check_column = df.columns[0]  # Use the first column if "Any" is selected

    if not display_column:
        display_column = df.columns[0]  # Use the first column if no display column is specified

    # Convert both check_column and display_column to the same data type as scanned_data
    df[check_column] = df[check_column].astype(type(scanned_data))
    df[display_column] = df[display_column].astype(type(scanned_data))

    # Find the rows where the check_column contains the scanned_data
    matching_rows = df[df[check_column] == scanned_data[1:]]

    # If matching rows are found, return the value of the display_column
    if not matching_rows.empty:
        return matching_rows[display_column].values[0]
    else:
        return "No matching data found"



# def extract_barcode_data(image):
#     st.write("Extracting barcode data...")

#     # Use OpenCV's QR code detector
#     qr_detector = cv2.QRCodeDetector()
#     data, bbox, _ = qr_detector.detectAndDecode(image)
#     st.write(f"QR Code Detection: Data: {data}, BBox: {bbox}")

#     if data:
#         return data  # Return the QR code data
    
#     # Use OpenCV's barcode detector
#     barcode_detector = cv2.barcode_BarcodeDetector()
#     retval, decoded_info, decoded_type = barcode_detector.detectAndDecode(image)
#     st.write(f"Barcode Detection: Retval: {retval}, Decoded Info: {decoded_info}, Decoded Type: {decoded_type}")

#     if retval:
#         return decoded_info[0] if decoded_info else None
    
#     # Fallback to pyzbar for additional barcode detection
#     decoded_objects = decode(image)
#     st.write(f"Decoded Objects (pyzbar): {decoded_objects}")
    
#     if decoded_objects:
#         barcode_data = decoded_objects[0].data.decode("utf-8")
#         return barcode_data
    
#     return None

# def camera_func():
#     # Create or get the SessionState
#     session_state = st.session_state
#     if 'scan_button_clicked' not in session_state:
#         session_state.scan_button_clicked = False

#     if st.button("Scan"):
#         session_state.scan_button_clicked = True
#         st.write("Please position the barcode in front of your phone's camera.")
            
#     if session_state.scan_button_clicked:
#         # Capture camera input
#         uploaded_image = st.camera_input("Scan QR code or barcode")
        
#         if uploaded_image is not None:
#             # Convert uploaded image to NumPy array
#             pil_image = Image.open(uploaded_image)
#             numpy_image = np.array(pil_image)

#             # Ensure the image is in RGB format
#             if len(numpy_image.shape) == 2 or numpy_image.shape[2] == 1:  # If grayscale, convert to RGB
#                 numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2RGB)

#             # Convert to grayscale
#             gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
#             st.image(gray, use_column_width=True, caption="Grayscale Image")

#             barcode_data = extract_barcode_data(gray)
#             st.write(f"Extracted Barcode Data: {barcode_data}")

#             if barcode_data:
#                 st.success(f"Barcode Data: {barcode_data}")
#                 session_state.scan_button_clicked = False
#                 return barcode_data
#             else:
#                 st.info("No QR code or barcode detected.")
#                 session_state.scan_button_clicked = False

#         stop_scanning = st.button("Stop Scanning")
#         if stop_scanning:
#             session_state.scan_button_clicked = False

#     return None
#------------------------
# def sharpen_image(image):
#     # Apply an unsharp mask to the image
#     gaussian_blur = cv2.GaussianBlur(image, (0, 0), 3)
#     sharpened = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)
#     return sharpened

# def extract_barcode_data(image):
#     # Use OpenCV's QR code detector as a fallback
#     qr_detector = cv2.QRCodeDetector()
#     data, bbox, _ = qr_detector.detectAndDecode(image)
    
#     if data:
#         return data  # Return the QR code data
    
#     return None

# def detect_qr_code(image):
#     qr_detector = cv2.QRCodeDetector()
#     data, bbox, _ = qr_detector.detectAndDecode(image)
#     st.write(f"QR Code Detection: Data: {data}, BBox: {bbox}")
#     return data, bbox

# def detect_and_crop_barcode(image):
#     st.write("Detecting barcode using contours...")

#     # Sharpen the image
#     sharpened_image = sharpen_image(image)
#     st.image(sharpened_image, use_column_width=True, caption="Sharpened Image")

#     # Convert the image to grayscale and apply edge detection
#     gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
#     resized_image = cv2.resize(gray, (128, 128))  # Adjust size based on your model
#     resized_image = resized_image / 255.0  # Normalize pixel values
#     edged = cv2.Canny(gray, 50, 200)

#     # Find contours in the edged image
#     contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Loop over the contours to find potential barcode regions
#     for contour in contours:
#         # Compute the bounding box of the contour and use it to draw the rectangle
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = w / float(h)
        
#         # Assume a barcode has a rectangular shape with an aspect ratio between 2 and 6
#         if 2 <= aspect_ratio <= 6 and w > 100 and h > 20:  # Adjust width and height thresholds as needed
#             cropped_image = sharpened_image[y:y+h, x:x+w]
#             st.image(cropped_image, caption="Cropped Image for Barcode Detection")
            
#             # Attempt to decode cropped region using pyzbar
#             decoded_objects = decode(resized_image)
#             if decoded_objects:
#                 barcode_data = decoded_objects[0].data.decode("utf-8")
#                 return barcode_data
    
#     return None

# def camera_func():
#     # Create or get the SessionState
#     session_state = st.session_state
#     if 'scan_button_clicked' not in session_state:
#         session_state.scan_button_clicked = False

#     if st.button("Scan"):
#         session_state.scan_button_clicked = True
#         st.write("Please position the barcode in front of your phone's camera.")
            
#     if session_state.scan_button_clicked:
#         # Capture camera input
#         uploaded_image = st.camera_input("Scan QR code or barcode")
        
#         if uploaded_image is not None:
#             # Convert uploaded image to NumPy array
#             pil_image = Image.open(uploaded_image)
#             numpy_image = np.array(pil_image)

#             # Convert to grayscale
#             gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
#             st.image(gray, use_column_width=True, caption="Grayscale Image")

#             # First, check for QR code
#             qr_data, qr_bbox = detect_qr_code(gray)
#             if qr_data:
#                 st.success(f"QR Code Data: {qr_data}")
#                 session_state.scan_button_clicked = False
#                 return qr_data

#             # If no QR code, check for barcode using contours
#             barcode_data = detect_and_crop_barcode(numpy_image)
#             if barcode_data:
#                 st.success(f"Barcode Data: {barcode_data}")
#                 session_state.scan_button_clicked = False
#                 return barcode_data

#             # If no QR code or barcode detected
#             st.info("No QR code or barcode detected.")
#             session_state.scan_button_clicked = False

#         stop_scanning = st.button("Stop Scanning")
#         if stop_scanning:
#             session_state.scan_button_clicked = False

#     return None

def camera_func():
    # Create or get the SessionState
    session_state = st.session_state
    if 'scan_button_clicked' not in session_state:
        session_state.scan_button_clicked = False

    if st.button("Scan"):
        session_state.scan_button_clicked = True
        st.write("Please upload an image of the barcode.")

    if session_state.scan_button_clicked:
        # Allow user to upload an image
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_image is not None:
            # Convert uploaded image to NumPy array
            pil_image = Image.open(uploaded_image)
            numpy_image = np.array(pil_image)

            # Convert to grayscale
            gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            st.image(gray, use_column_width=True, caption="Grayscale Image")

            # Decode barcode using pyzbar
            decoded_objects = decode(gray)
            if decoded_objects:
                scanned_data = decoded_objects[0].data.decode("utf-8")
                session_state.scan_button_clicked = False
                return scanned_data
            else:
                st.info("No barcode detected in the uploaded image.")
                session_state.scan_button_clicked = False

        stop_scanning = st.button("Stop Scanning")
        if stop_scanning:
            session_state.scan_button_clicked = False

    return None

def main():
    st.set_page_config(page_title="Table Extraction with OCR", layout="wide")
                
    # Sidebar options
    st.sidebar.header("Operations")
    operation = st.sidebar.radio("Select an operation", ["Home","Retrieve Table", "Delete Table", "Edit Table", "Scan to Display"])

    # Fetch list of table names from the database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [row[0] for row in cursor.fetchall()]
    cursor.close()
    

    if operation == "Home":
        st.title("Table Extraction with OCR")

        # File upload options
        uploaded_file = st.file_uploader("Upload an image, CSV, or Excel file", type=["jpg", "jpeg", "png", "csv", "xlsx"])

        # Initialize session state for DataFrame and formula
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'manipulated_df' not in st.session_state:
            st.session_state.manipulated_df = None
        if 'formula' not in st.session_state:
            st.session_state.formula = ""

       
        if uploaded_file is not None:
            if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                # Process image and extract DataFrame
                image = Image.open(uploaded_file).convert("RGB")
                enhancer = ImageEnhance.Sharpness(image)
                enhanced_image = enhancer.enhance(2)  # Adjust the factor as needed to control sharpness
                enhanced_image_np = np.array(enhanced_image)

                # Load the detection model and feature extractor
                detection_model, detection_feature_extractor = load_table_transformer_detection_model()
                detection_outputs = process_image_with_model(enhanced_image_np, detection_feature_extractor, detection_model)
                cropped_image = enhanced_image

                # Resize the cropped image
                width, height = cropped_image.size
                resized_image = cropped_image.resize((int(width * 1), int(height * 1)))
                resized_image_np = np.array(resized_image)

                # Perform OCR on the resized image
                ocr = load_ppocr_model()
                ocr_results = ocr.ocr(resized_image_np)

                # Load the structure recognition model and feature extractor
                structure_model, structure_feature_extractor = load_table_transformer_structure_model()
                if structure_model is None or structure_feature_extractor is None:
                    st.error("Failed to load the Table Transformer Structure Recognition model.")
                    return
                structure_outputs = process_image_with_model(resized_image_np, structure_feature_extractor, structure_model)

                # Process the outputs for structure recognition
                target_sizes = [image.size[::-1]]
                results = structure_feature_extractor.post_process_object_detection(structure_outputs, threshold=0.7, target_sizes=[(height, width)])[0]
                label_dict = structure_model.config.id2label
                labels_boxes_dict = {}

                for label, box in zip(results['labels'], results['boxes']):
                    label_name = label_dict[label.item()]
                    if label_name not in labels_boxes_dict:
                        labels_boxes_dict[label_name] = []
                    labels_boxes_dict[label_name].append(box.tolist())

                column_boxes = labels_boxes_dict.get('table column', [])
                row_boxes = labels_boxes_dict.get('table row', [])

                df = extract_columns_and_display_excel(ocr_results, resized_image, row_boxes, column_boxes)
                st.session_state.df = df  # Store the DataFrame in the session state
                st.subheader("Extracted DataFrame")
                st.dataframe(df)

            elif uploaded_file.type == "text/csv":
                try:
                    df = pd.read_csv(uploaded_file, index_col=None)
                    st.session_state.df = df
                    st.subheader("Uploaded CSV File")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error processing CSV file: {e}")

            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                try:
                    df = pd.read_excel(uploaded_file, index_col=None)
                    st.session_state.df = df
                    st.subheader("Uploaded Excel File")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error processing Excel file: {e}")

            # Save DataFrame to Database
            st.subheader("Save DataFrame to Database")
            table_name_to_save = st.text_input("Enter table name to save")
            if st.button("Save DataFrame"):
                if table_name_to_save:
                    df_to_save = st.session_state.df
                    if df_to_save is not None:
                        save_dataframe_to_sqlite(df_to_save, table_name_to_save)
                    else:
                        st.error("No DataFrame to save.")

    elif operation == "Retrieve Table":
        table_name_to_retrieve = st.selectbox("Select Table Name to Retrieve", options=table_names)
        if table_name_to_retrieve:
            if table_exists(table_name_to_retrieve):
                retrieved_df = retrieve_dataframe_from_sqlite(table_name_to_retrieve)
                st.subheader(f"Retrieved DataFrame from '{table_name_to_retrieve}'")
                st.dataframe(retrieved_df)
                st.session_state.df = retrieved_df

    elif operation == "Delete Table":
        table_name_to_delete = st.selectbox("Select Table Name to Delete", options=table_names)
        if table_name_to_delete:
            if st.button("Delete Table"):
                delete_table_from_sqlite(table_name_to_delete)
    
    elif operation == "Edit Table":
       
        table_name_to_edit = st.selectbox("Select Table Name to Edit", options=table_names)
        if table_name_to_edit and table_exists(table_name_to_edit):
            # Retrieve DataFrame from the selected table
            df_to_edit = retrieve_dataframe_from_sqlite(table_name_to_edit)
            st.subheader(f"Editing DataFrame from '{table_name_to_edit}'")
            st.dataframe(df_to_edit)

            # Input Formula
            st.subheader("Input Formula")
            st.session_state.formula = st.text_input("Enter Formula", value=st.session_state.formula)

            # Apply Formula Button
            if st.button("Apply Formula"):
                if st.session_state.formula:
                    # Apply the formula to the DataFrame
                    manipulated_df = apply_formula_to_dataframe(df_to_edit.copy(), st.session_state.formula)
                    if manipulated_df is not None:
                        # Display the manipulated DataFrame
                        st.subheader("Manipulated DataFrame")
                        st.dataframe(manipulated_df)
                        st.session_state.manipulated_df = manipulated_df
                        

            # Save Options
            st.subheader("Save Options")

            # Save as New Table Section
           
            save_option = st.radio("Choose a save option:", ("Save as New Table", "Overwrite Existing Table"))

            if save_option == "Save as New Table":
                new_table_name = st.text_input("Enter New Table Name", key="new_table_name")
                save= st.button("Save")
                if new_table_name and save:
                    # Save the manipulated DataFrame as a new table
                    save_dataframe_to_sqlite(st.session_state.manipulated_df, new_table_name)
                    

            # Overwrite Existing Table Section
            elif save_option == "Overwrite Existing Table" and st.button("Save"):
                # Save the manipulated DataFrame to the existing table
                save_dataframe_to_sqlite(st.session_state.manipulated_df, table_name_to_edit, overwrite=True)
                




    elif operation == "Scan to Display":
        table_name_to_scan = st.selectbox("Select Table Name", options=table_names)
        if table_name_to_scan:
            if table_exists(table_name_to_scan):
                df_to_scan = retrieve_dataframe_from_sqlite(table_name_to_scan)
                st.subheader(f"Scanning DataFrame from '{table_name_to_scan}'")
                st.dataframe(df_to_scan)
                st.session_state.df = df_to_scan

                search_options = ["Any"] + list(st.session_state.df.columns)
                display_options = list(st.session_state.df.columns)

                selected_search_column = st.selectbox("Select Column to Search", options=search_options)
                selected_display_column = st.selectbox("Select Column to Display", options=display_options)

                scanned_data = camera_func()
                if scanned_data:
                    st.success(f"Scanned Data: {scanned_data}")
                    result = check_databases(df_to_scan, scanned_data, selected_search_column, selected_display_column)
                    st.subheader("Database Check Result")
                    st.write(f"Value: {result}")

if __name__ == "__main__":
    main()
