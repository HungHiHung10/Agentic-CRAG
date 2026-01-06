# modules/ingestion.py
import re
import os
import shutil
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import CHROMA_DB_DIR, FILES


def split_noi_khoa_gs_chau():
    print("[PROCESS] Processing Noi Khoa YHCT - GS Hoang Bao Chau...")
    path = FILES["noi_khoa_gs_chau"]
    
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return []

    loader = Docx2txtLoader(path)
    raw_text = loader.load()[0].page_content
    raw_text = re.sub(r'\n*\s*_{8,}\s*\n*', '\n__________\n', raw_text)
    parts = re.split(r'\n__________\n', raw_text)

    docs = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part: continue
        lines = [line.strip() for line in part.split('\n') if line.strip()]
        title_line = lines[0] if lines else f"Untitled {i}"
        for line in lines:
            if line.isupper() or (len(line) > 5 and line == line.upper()):
                title_line = line; break
        docs.append(Document(page_content=part, metadata={"title": title_line.strip()}))

    PATTERNS = [r'^[A-Z]\.\s', r'^\d+\.\s', r'^\d+\.\d+\.\s', r'^[a-zđ]\.\s']

    def recursive_split_gs(lines, current_title, current_section, pattern_index):
        if pattern_index >= len(PATTERNS):
            content = '\n'.join(lines).strip()
            if not content: return []
            return [Document(
                page_content=content,
                metadata={
                    "source": "Noi_Khoa_YHCT_GS_Hoang_Bao_Chau",
                    "title": current_title,
                    "section": current_section
                }
            )]

        current_pattern = PATTERNS[pattern_index]
        splits = []
        for i, line in enumerate(lines):
            if re.match(current_pattern, line.strip()):
                splits.append((i, line.strip()))

        if not splits:
            return recursive_split_gs(lines, current_title, current_section, pattern_index + 1)

        results = []
        splits = [(0, None)] + splits + [(len(lines), None)]
        splits.sort(key=lambda x: x[0])

        for j in range(len(splits) - 1):
            start_idx, header = splits[j]
            end_idx = splits[j + 1][0]
            sub_lines = lines[start_idx + 1 : end_idx]

            if header is None:
                if not sub_lines: continue
                suffix = " (Dẫn nhập)" if "Dẫn nhập" not in current_section else ""
                new_section_name = f"{current_section}{suffix}"
                results.extend(recursive_split_gs(sub_lines, current_title, new_section_name, pattern_index + 1))
            else:
                new_section_name = f"{current_section} - {header}"
                content_for_next_level = [header] + sub_lines
                results.extend(recursive_split_gs(content_for_next_level, current_title, new_section_name, pattern_index + 1))
        return results

    chunks = []
    main_headers = ["ĐẠI CƯƠNG", "CHỨNG TRỊ", "NGUYÊN NHÂN"]

    for doc in docs:
        part = doc.page_content.strip()
        title = doc.metadata.get("title", "")
        if not part: continue
        lines = part.split('\n')
        
        splits = []
        for i, line in enumerate(lines):
            clean_header = line.strip().rstrip(':')
            if clean_header in main_headers:
                splits.append((i, clean_header))
        
        splits = [(0, None)] + splits + [(len(lines), None)]
        splits.sort(key=lambda x: x[0])

        for j in range(len(splits) - 1):
            start_index, header = splits[j]
            end_index = splits[j + 1][0]
            if header is None: continue
            
            section_lines = lines[start_index + 1 : end_index]
            if not section_lines: continue
            
            sub_docs = recursive_split_gs(section_lines, title, header, pattern_index=0)
            for sub_doc in sub_docs:
                Content = f"{header}\n\n{sub_doc.page_content}"
                sub_doc.page_content = Content
                chunks.append(sub_doc)
                
    return chunks

def split_benh_ngu_quan():
    print("[PROCESS] Processing Benh Ngu Quan...")
    file_path = FILES["benh_ngu_quan"]
    
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return []

    BOOK_NAME = "bệnh ngũ quan"
    MARKER_START = "CHÍN (9) ĐIỀU CĂN DẶN"
    NO_SPLIT_TITLES = ["CHÍN (9) ĐIỀU CĂN DẶN", "LỜI NÓI ĐẦU"]

    loader = Docx2txtLoader(file_path)
    raw_text = loader.load()[0].page_content
    raw_text = raw_text.replace('</break>', ' ').replace('<break>', ' ')
    raw_text = re.sub(r'\n*\s*_{8,}\s*\n*', '\n__________\n', raw_text)
    parts = re.split(r'\n__________\n', raw_text)

    SUB_PATTERNS = [
        r'^\s*(?:Phần|Chương)?\s*[IVX]+\.[\s\t]*',
        r'^\s*\d+\.[\s\t]*',
        r'^\s*\d+\.\d+\.[\s\t]*',
        r'^\s*[a-zđ]\.[\s\t]*'
    ]

    def recursive_split_nq(lines, current_title, current_section_name, pattern_idx):
        def create_doc(content_lines, section_override=None):
            content = '\n'.join(content_lines).strip()
            if not content: return None
            final_section = section_override if section_override else current_section_name
            full_content = f"{current_title}-{final_section}\nNỘI DUNG:\n{content}"
            return Document(
                page_content=full_content,
                metadata={"source": BOOK_NAME, "title": current_title, "section": final_section}
            )

        if pattern_idx >= len(SUB_PATTERNS):
            doc = create_doc(lines)
            return [doc] if doc else []

        current_pattern = SUB_PATTERNS[pattern_idx]
        splits = []

        for i, line in enumerate(lines):
            line_strip = line.strip()
            is_match = False
            if re.match(current_pattern, line_strip): is_match = True
            
            if not is_match and pattern_idx == 0:
                if any(line_strip.upper().startswith(k) for k in ["PHỤ LỤC", "PHỤ PHƯƠNG"]): is_match = True
            
            if not is_match and pattern_idx == 1:
                keywords_rescue = ["Phép chữa", "Phương pháp", "Điều trị", "Triệu chứng"]
                if any(line_strip.startswith(k) for k in keywords_rescue) and len(line_strip) < 50: is_match = True

            if is_match: splits.append((i, line_strip))

        if not splits:
            if pattern_idx == 0:
                doc = create_doc(lines, section_override="Tổng quan")
                return [doc] if doc else []
            else:
                return recursive_split_nq(lines, current_title, current_section_name, pattern_idx + 1)

        results = []
        splits = [(0, None)] + splits + [(len(lines), None)]
        splits.sort(key=lambda x: x[0])

        for j in range(len(splits) - 1):
            start_idx, header = splits[j]
            end_idx = splits[j + 1][0]
            sub_lines = lines[start_idx + 1 : end_idx]
            if not sub_lines: continue

            if header is None:
                suffix = " (Dẫn nhập)" if "Dẫn nhập" not in current_section_name else ""
                final_sec = f"{current_section_name}{suffix}" if current_section_name else "Tổng quan"
                doc = create_doc(sub_lines, section_override=final_sec)
                if doc: results.append(doc)
            else:
                new_section = header if not current_section_name else f"{current_section_name} > {header}"
                special_list = ["PHỤ LỤC", "PHỤ PHƯƠNG", "BÀI ĐỌC THÊM"]
                if any(k in header.upper() for k in special_list):
                    results.extend(recursive_split_nq(sub_lines, current_title, new_section, 0))
                else:
                    content_next = [header] + sub_lines
                    results.extend(recursive_split_nq(content_next, current_title, new_section, pattern_idx + 1))
        return results

    chunks = []
    
    def process_buffer(lines, title):
        if not lines: return
        if any(k in title.upper() for k in NO_SPLIT_TITLES):
            full = f"{title}-Toàn văn\nNỘI DUNG:\n" + "\n".join(lines)
            chunks.append(Document(page_content=full, metadata={"source": BOOK_NAME, "title": title, "section": "Toàn văn"}))
            return
        chunks.extend(recursive_split_nq(lines, title, "", pattern_idx=0))

    is_phase_2 = False
    pre_content = []
    active_title = "Tổng quan"
    last_processed_title = "Tổng quan"
    current_disease_buffer = []

    all_lines = []
    for part in parts:
        all_lines.extend([line.strip() for line in part.split('\n') if line.strip()])

    for i, line in enumerate(all_lines):
        line_strip = line.strip()
        if not line_strip: continue

        if not is_phase_2:
            if MARKER_START in line_strip.upper() and not re.search(r'\d+$', line_strip):
                is_phase_2 = True
                if pre_content:
                    chunks.append(Document(page_content=f"PHẦN MỞ ĐẦU / MỤC LỤC\n\n" + "\n".join(pre_content), 
                                           metadata={"source": BOOK_NAME, "title": BOOK_NAME, "section": "Intro"}))
                active_title = line_strip
                last_processed_title = line_strip
                current_disease_buffer = []
                continue
            pre_content.append(line_strip)
            continue

        is_upper = (line_strip.isupper() and len(line_strip) > 5
                    and not re.match(SUB_PATTERNS[0], line_strip)
                    and not any(bad in line_strip.upper() for bad in ["TRƯỜNG ĐẠI HỌC", "KHOA Y HỌC", "BỘ MÔN", "BỆNH NGŨ QUAN"]))
        is_appendix = any(k in line_strip.upper() for k in ["PHỤ LỤC", "PHỤ PHƯƠNG"])

        if is_upper:
            process_buffer(current_disease_buffer, active_title)
            current_disease_buffer = []
            if is_appendix:
                active_title = f"{last_processed_title} - {line_strip}"
            else:
                active_title = line_strip
                last_processed_title = line_strip
            continue
        current_disease_buffer.append(line_strip)

    process_buffer(current_disease_buffer, active_title)
    return chunks

def split_nhi_khoa():
    print("[PROCESS] Processing Nhi Khoa YHCT...")
    file_path = FILES["nhi_khoa"]
    
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return []

    raw_text = Docx2txtLoader(file_path).load()[0].page_content

    def is_skip(line: str) -> bool:
        line = line.strip()
        return (not line or line.lower() == "</break>" or re.fullmatch(r"_+", line) is not None)

    lines = [line.strip() for line in raw_text.split("\n") if not is_skip(line)]

    CHAPTER_PATTERN = re.compile(r"^CHƯƠNG\s+([IVXLC]+)", re.I)
    LESSON_PATTERN  = re.compile(r"^Bài\s+\d+", re.I)
    ROMAN_HEADER    = re.compile(r"^[IVX]+\.\s+.+")
    NUMBER_HEADER   = re.compile(r"^(\d+(?:\.\d+)*\.)\s+(.+)")

    def parse_number_path(header: str):
        nums = header.split()[0].strip(".")
        return [int(x) for x in nums.split(".")]

    chunks = []
    current_chapter = None
    in_chapter = False
    current_lesson = None
    current_title = None
    section_stack = []
    content_buffer = []

    def flush_buffer():
        if not content_buffer or not section_stack: return
        full_section = " > ".join(h for _, h in section_stack)
        leaf_header = section_stack[-1][1]
        chunks.append(Document(
            page_content="\n".join([leaf_header] + content_buffer).strip(),
            metadata={
                "source": "nhi-khoa-y-hoc-co-truyen",
                "title": current_title,
                "section": full_section
            }
        ))
        content_buffer.clear()

    for line in lines:
        chap_match = CHAPTER_PATTERN.match(line)
        if chap_match:
            flush_buffer()
            current_chapter = chap_match.group(1)
            in_chapter = True
            current_lesson = None; current_title = None; section_stack.clear()
            continue

        if not in_chapter: continue

        if LESSON_PATTERN.match(line):
            flush_buffer()
            section_stack.clear()
            current_lesson = line
            current_title = None
            continue

        if current_lesson and not current_title and line.isupper():
            current_title = line
            continue

        if not current_title: continue

        if ROMAN_HEADER.match(line):
            flush_buffer()
            section_stack = [(1, line)]
            continue

        num_match = NUMBER_HEADER.match(line)
        if num_match:
            if current_chapter == "I": continue
            flush_buffer()
            level = len(parse_number_path(line)) + 1
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, line))
            continue

        if section_stack:
            content_buffer.append(line)

    flush_buffer()
    return chunks

def split_noi_khoa_general():
    print("[PROCESS] Processing Noi Khoa General...")
    file_path = FILES["noi_khoa_general"]
    
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return []

    loader = Docx2txtLoader(file_path)
    raw_text = loader.load()[0].page_content
    chapter_pattern = r'(?m)^(?![0-9]+\.|I+\.|V\.)(?![a-z])([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ\s,()-]{3,})$'
    raw_text_processed = re.sub(chapter_pattern, r'\n__________\n\1', raw_text)
    parts = re.split(r'\n__________\n', raw_text_processed)

    docs_by_section = []
    for part in parts:
        part = part.strip()
        lines = [line.strip() for line in part.split('\n') if line.strip()]
        if not lines or len(part) < 50: continue
        title_line = lines[0]
        docs_by_section.append(Document(page_content=part, metadata={"title": title_line}))

    SUB_PATTERNS = [
        r'^\d+\.\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ]',
        r'^\d+\.\d+\.\s',
        r'^\d+\.\d+\.\d+\.\s',
        r'^[a-zđ]\.\s',
    ]

    def recursive_split_gen(lines, current_title, current_section_name, pattern_idx):
        if pattern_idx >= len(SUB_PATTERNS):
            content = '\n'.join(lines).strip()
            if not content: return []
            return [Document(
                page_content=content,
                metadata={
                    "disease": current_title,
                    "section": current_section_name,
                    "source": "noi-khoa-y-hoc-co-truyen"
                }
            )]

        current_pattern = SUB_PATTERNS[pattern_idx]
        splits = []
        for i, line in enumerate(lines):
            if re.match(current_pattern, line.strip()):
                splits.append((i, line.strip()))

        if not splits:
            return recursive_split_gen(lines, current_title, current_section_name, pattern_idx + 1)

        results = []
        splits = [(0, None)] + splits + [(len(lines), None)]

        for j in range(len(splits) - 1):
            start_idx, header = splits[j]
            end_idx = splits[j + 1][0]
            sub_lines = lines[start_idx + 1 : end_idx]

            if header is None:
                if not sub_lines: continue
                suffix = " (Dẫn nhập)" if "Dẫn nhập" not in current_section_name else ""
                new_section_name = f"{current_section_name}{suffix}"
                results.extend(recursive_split_gen(sub_lines, current_title, new_section_name, pattern_idx + 1))
            else:
                new_section_name = f"{current_section_name} > {header}"
                content_for_next_level = [header] + sub_lines
                results.extend(recursive_split_gen(content_for_next_level, current_title, new_section_name, pattern_idx + 1))
        return results

    refined_docs = []
    for doc in docs_by_section:
        part = doc.page_content.strip()
        title = doc.metadata.get("title", "")
        lines = part.split('\n')
        refined_docs.extend(recursive_split_gen(lines, title, title, pattern_idx=0))

    chunks = [d for d in refined_docs if "</break>" not in d.metadata.get('disease', '')]
    return chunks

def create_vector_db(embedding_model, force_rebuild=False):
    if os.path.exists(CHROMA_DB_DIR) and not force_rebuild:
        print(f"[INFO] ChromaDB already exists at {CHROMA_DB_DIR}. Loading...")
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model, collection_name='y_hoc_co_truyen')

    if force_rebuild and os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print("[SUCCESS] Removed old Chroma directory")

    all_chunks = []
    all_chunks.extend(split_noi_khoa_gs_chau())
    all_chunks.extend(split_benh_ngu_quan())
    all_chunks.extend(split_nhi_khoa())
    all_chunks.extend(split_noi_khoa_general())
    
    if not all_chunks:
        print("[WARNING] No chunks created. Check file paths.")
        return None

    print(f"[PROCESS] Creating Vector DB with {len(all_chunks)} chunks...")
    chroma_db = Chroma.from_documents(
        documents=all_chunks,
        collection_name='y_hoc_co_truyen',
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=CHROMA_DB_DIR
    )
    print(f"[SUCCESS] Vector DB Created with count: {chroma_db._collection.count()}")
    return chroma_db