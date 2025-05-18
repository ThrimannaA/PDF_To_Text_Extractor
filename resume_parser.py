import streamlit as st
import re
import spacy
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from collections import OrderedDict
import os

# ========== STREAMLIT CONFIGURATION (MUST BE FIRST) ==========
st.set_page_config(
    page_title="Resume Parser & Accuracy Checker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== FUNCTION DEFINITIONS ==========
@st.cache_resource
def load_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading language model... (this may take a few minutes)")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_nlp()

def preprocess_image(image):
    """Enhanced image preprocessing for better OCR"""
    img = image.convert('L')  # Convert to grayscale
    img = ImageOps.autocontrast(img, cutoff=5)  # Improve contrast
    img = ImageEnhance.Sharpness(img).enhance(2.0)  # Enhance sharpness
    return img

def detect_sections(text):
    """Robust section detection with fixed regex patterns"""
    section_headers = [
        'education', 'academic background', 'qualifications',
        'experience', 'work experience', 'employment',
        'projects', 'personal projects',
        'technical skills', 'skills', 'key skills',
        'contact', 'contact information',
        'references', 'hackathons',
        'extracurricular activities', 'activities'
    ]

    sections = OrderedDict()
    current_section = "HEADER"
    sections[current_section] = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        is_section = False
        for header in section_headers:
            pattern = r'^\s*' + re.escape(header) + r'\s*:?\s*$'
            if re.search(pattern, line, re.IGNORECASE):
                current_section = header.upper()
                sections[current_section] = []
                is_section = True
                break

        if not is_section:
            sections[current_section].append(line)

    return sections

def format_sections(sections):
    """Strict formatting with clear section separation"""
    formatted_text = []

    for section, content in sections.items():
        if not content:
            continue

        if section != "HEADER":
            formatted_text.append(f"\n\n{section.upper()}\n\n")
        else:
            if content:
                formatted_text.append(f"{content[0].upper()}\n\n")
                if len(content) > 1:
                    formatted_text.append("\n".join(content[1:]) + "\n")
            continue

        if 'contact' in section.lower():
            contact_info = []
            for line in content:
                if re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', line):
                    contact_info.append(f"Phone: {line}")
                elif '@' in line:
                    contact_info.append(f"Email: {line}")
                elif 'linkedin.com' in line.lower():
                    contact_info.append(f"LinkedIn: {line}")
                elif 'github.com' in line.lower():
                    contact_info.append(f"GitHub: {line}")
                else:
                    contact_info.append(line)
            formatted_text.append("\n".join(contact_info))
        else:
            formatted_text.append("\n".join([f"• {line}" for line in content]))

    return "".join(formatted_text).strip()

def extract_text_with_structure(pdf_path):
    """Robust text extraction with error handling"""
    try:
        text = extract_text(pdf_path)
        if len(text.strip().split()) > 50:
            sections = detect_sections(text)
            return format_sections(sections)
    except Exception as e:
        st.warning(f"PDFMiner extraction attempt failed: {str(e)}")

    try:
        images = []
        try:
            images = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            st.warning(f"PDF conversion warning: {str(e)}")
            images = convert_from_path(pdf_path, dpi=200)

        full_text = []
        for img in images:
            try:
                processed_img = preprocess_image(img)
                text = pytesseract.image_to_string(processed_img, config='--psm 6')
                full_text.append(text)
            except Exception as e:
                st.warning(f"OCR processing warning: {str(e)}")
                continue

        if full_text:
            combined_text = '\n'.join(full_text)
            sections = detect_sections(combined_text)
            return format_sections(sections)
        return "OCR completed but no text extracted"
    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        return "Extraction failed"

def normalize_text(text):
    """Normalize text for comparison"""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[^\w\s]', '', text)
    return text

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_accuracy(extracted_text, ground_truth_text):
    """Calculate accuracy percentage"""
    extracted_norm = normalize_text(extracted_text)
    ground_truth_norm = normalize_text(ground_truth_text)
    
    if not extracted_norm or not ground_truth_norm:
        return 0.0, extracted_norm, ground_truth_norm
    
    distance = levenshtein_distance(extracted_norm, ground_truth_norm)
    max_length = max(len(extracted_norm), len(ground_truth_norm))
    
    accuracy = (1 - distance / max_length) * 100 if max_length > 0 else 0.0
    return round(accuracy, 2), extracted_norm, ground_truth_norm, distance, max_length

# ========== MAIN APP ==========
def main():
    st.title("📄 Resume Parser & Accuracy Checker")
    st.markdown("Upload a resume PDF and optionally a ground truth text file to check extraction accuracy.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resume PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
    with col2:
        st.subheader("Upload Ground Truth (Optional)")
        uploaded_truth = st.file_uploader("Choose a text file", type="txt", key="truth_uploader")
    
    if uploaded_pdf:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
        
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        with st.spinner("Extracting text from resume..."):
            extracted_text = extract_text_with_structure(temp_pdf_path)
        
        # Display extracted and ground truth side by side
        st.subheader("Resume Content Comparison")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**Extracted Text**")
            st.text_area("", extracted_text, height=400, key="extracted_text", label_visibility="collapsed")
        
        if uploaded_truth:
            temp_truth_path = os.path.join(temp_dir, uploaded_truth.name)
            with open(temp_truth_path, "wb") as f:
                f.write(uploaded_truth.getbuffer())
            
            with open(temp_truth_path, "r", encoding='utf-8') as f:
                ground_truth_text = f.read()
            
            with col_right:
                st.markdown("**Ground Truth Text**")
                st.text_area("", ground_truth_text, height=400, key="ground_truth", label_visibility="collapsed")
        
        # Download button for extracted text
        output_file = os.path.splitext(uploaded_pdf.name)[0] + "_extracted.txt"
        output_path = os.path.join(temp_dir, output_file)
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(extracted_text)
        
        with open(output_path, "r", encoding='utf-8') as f:
            st.download_button(
                label="Download Extracted Text",
                data=f,
                file_name=output_file,
                mime="text/plain"
            )
        
        if uploaded_truth:
            # Calculate accuracy
            accuracy, extracted_norm, ground_truth_norm, distance, max_length = calculate_accuracy(
                extracted_text, ground_truth_text)
            
            # Accuracy metrics
            st.subheader("Accuracy Analysis")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Accuracy Score", f"{accuracy}%")
            with metric_col2:
                st.metric("Levenshtein Distance", distance)
            with metric_col3:
                st.metric("Max Length (normalized)", max_length)
            
            # Normalized comparison
            st.subheader("Normalized Text Comparison")
            norm_col1, norm_col2 = st.columns(2)
            with norm_col1:
                st.markdown("**Normalized Extracted Text**")
                st.text_area("", 
                            extracted_norm[:5000] + ("..." if len(extracted_norm) > 5000 else ""), 
                            height=300, 
                            label_visibility="collapsed")
            with norm_col2:
                st.markdown("**Normalized Ground Truth**")
                st.text_area("", 
                            ground_truth_norm[:5000] + ("..." if len(ground_truth_norm) > 5000 else ""), 
                            height=300,
                            label_visibility="collapsed")

if __name__ == "__main__":
    main()