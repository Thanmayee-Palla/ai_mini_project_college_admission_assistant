"""
ingest.py — College Admission Assistant
Reads college_dataset_350.csv, builds rich typed chunks per college,
embeds them with SentenceTransformers (MiniLM) and stores in ChromaDB.
Run once before starting the FastAPI server:  python ingest.py
"""

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "college_dataset_350.csv")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "colleges"


def _rank_band(rank: int) -> str:
    if rank <= 1_000:   return "under 1000"
    if rank <= 5_000:   return "1000-5000"
    if rank <= 10_000:  return "5000-10000"
    if rank <= 25_000:  return "10000-25000"
    return "above 25000"


def _fee_label(fee: float) -> str:
    if fee < 50_000:    return "low (under Rs.50,000)"
    if fee < 150_000:   return "moderate (Rs.50k-Rs.1.5L)"
    if fee < 300_000:   return "high (Rs.1.5L-Rs.3L)"
    return "very high (above Rs.3L)"


def _rating_label(r: float) -> str:
    if r >= 4.5: return "excellent"
    if r >= 4.0: return "very good"
    if r >= 3.5: return "good"
    if r >= 3.0: return "average"
    return "below average"


def _top_aspects(placement, academics, infra, faculty) -> list:
    aspects = {"Placements": placement, "Academics": academics,
               "Infrastructure": infra, "Faculty": faculty}
    return sorted(aspects, key=aspects.get, reverse=True)[:2]


def _make_chunk(college_name: str, chunk_type: str, text: str, row: pd.Series) -> dict:
    cid = f"{college_name}_{chunk_type}".replace(" ", "_").lower()
    return {
        "id": cid,
        "text": text,
        "metadata": {
            "college_name": college_name,
            "chunk_type": chunk_type,
            "state": str(row.get("State", "")),
            "district": str(row.get("District", "")),
        }
    }


def build_chunks_for_group(college_name: str, group: pd.DataFrame) -> list:
    chunks = []
    row0 = group.iloc[0]

    location_parts = group[["District", "State"]].drop_duplicates()
    locations = "; ".join(f"{r['District']}, {r['State']}" for _, r in location_parts.iterrows())

    streams  = group["Preferred_Stream"].dropna().unique().tolist()
    courses  = group["Preferred_Course"].dropna().unique().tolist()
    exams    = group["Entrance_Exam_Name"].dropna().unique().tolist()
    avg_rating = group["Ratings"].mean()
    ug_fees  = group["UG_Fee"].dropna()
    pg_fees  = group["PG_Fee"].dropna()

    # Overview
    overview_text = (
        f"{college_name} is located in {locations}. "
        f"It offers programmes in streams: {', '.join(streams)}. "
        f"Courses available: {', '.join(courses)}. "
        f"Entrance exams accepted: {', '.join(exams)}. "
        f"Overall rating: {avg_rating:.1f}/5 ({_rating_label(avg_rating)}). "
        f"UG fee range: Rs.{int(ug_fees.min()):,} to Rs.{int(ug_fees.max()):,}. "
        f"PG fee range: Rs.{int(pg_fees.min()):,} to Rs.{int(pg_fees.max()):,}."
    )
    chunks.append(_make_chunk(college_name, "overview", overview_text, row0))

    # Fees
    fee_lines = []
    for stream, sg in group.groupby("Preferred_Stream"):
        ug = sg["UG_Fee"].mean()
        pg = sg["PG_Fee"].mean()
        fee_lines.append(
            f"{stream} - UG avg Rs.{int(ug):,} ({_fee_label(ug)}), "
            f"PG avg Rs.{int(pg):,} ({_fee_label(pg)})"
        )
    scholarship_pct = (group["Scholarship_Status"] == "Yes").mean() * 100
    fee_text = (
        f"Fee structure at {college_name}: " + "; ".join(fee_lines) + ". "
        f"Scholarships are available to approximately {scholarship_pct:.0f}% of students."
    )
    chunks.append(_make_chunk(college_name, "fees", fee_text, row0))

    # Eligibility + Cutoff
    cutoff_lines = []
    for (exam, course), sg in group.groupby(["Entrance_Exam_Name", "Preferred_Course"]):
        ranks = sg["Entrance_Exam_Rank"].dropna()
        pct   = sg["12th_Percentage"].dropna()
        if ranks.empty:
            continue
        r_min, r_max = int(ranks.min()), int(ranks.max())
        if r_min == r_max:
            rank_str = f"rank {r_min} ({_rank_band(r_min)})"
        else:
            rank_str = f"rank {r_min}-{r_max} ({_rank_band(r_min)} to {_rank_band(r_max)})"

        cutoff_lines.append(
            f"{course} via {exam}: {rank_str}, "
            f"12th ~{pct.mean():.1f}%"
        )
    elig_text = (
        f"Eligibility and cutoff ranks at {college_name}: "
        + ("; ".join(cutoff_lines) if cutoff_lines else "check official website for cutoffs.")
    )
    chunks.append(_make_chunk(college_name, "eligibility", elig_text, row0))
    chunks.append(_make_chunk(college_name, "cutoff", elig_text.replace("Eligibility and cutoffs", "Cutoff ranks"), row0))

    # Admission process
    date_sample = group[["Application_Start_Date", "Application_End_Date"]].dropna().head(3)
    date_notes = ""
    if not date_sample.empty:
        starts = date_sample["Application_Start_Date"].unique()
        ends   = date_sample["Application_End_Date"].unique()
        date_notes = (
            f"Application start dates include: {', '.join(str(d) for d in starts[:3])}. "
            f"Application end dates include: {', '.join(str(d) for d in ends[:3])}. "
        )
    admission_text = (
        f"Admission process at {college_name}: "
        f"Step 1 - Appear for the required entrance exam ({', '.join(exams)}). "
        f"Step 2 - Register on the college admission portal. "
        f"Step 3 - Fill course preferences / web options. "
        f"Step 4 - Attend counselling and seat allotment. "
        f"Step 5 - Report to college with required documents and pay fees. "
        + date_notes
        + f"Scholarships available to ~{scholarship_pct:.0f}% of students based on merit/category."
    )
    chunks.append(_make_chunk(college_name, "admission", admission_text, row0))

    # Documents
    stream_docs = {
        "Engineering": "JEE/CET scorecard, 12th marksheet (PCM), Aadhar card, category certificate, passport photos",
        "Medical":     "NEET scorecard, 12th marksheet (PCB), medical fitness certificate, Aadhar card, photos",
        "Commerce":    "CUET/CET scorecard, 12th marksheet, Aadhar card, bank details for fees, photos",
        "Arts":        "CUET/merit scorecard, 12th marksheet, Aadhar card, TC, photos",
    }
    doc_lines = [f"{s}: {stream_docs.get(s, 'check official portal')}" for s in streams]
    doc_text = (
        f"Documents required for {college_name} admission: "
        + "; ".join(doc_lines) + ". "
        "Always carry originals plus 2 photocopies of every document."
    )
    chunks.append(_make_chunk(college_name, "documents", doc_text, row0))

    # Placements & ratings
    placement_avg = group["Placement"].mean()
    academic_avg  = group["Academics"].mean()
    infra_avg     = group["Infrastructure"].mean()
    faculty_avg   = group["Faculty"].mean()
    social_avg    = group["Social_Life"].mean()
    accomm_avg    = group["Accommodation"].mean()
    placements_text = (
        f"Placement and ratings at {college_name}: "
        f"Overall rating {avg_rating:.1f}/5. "
        f"Placement score {placement_avg:.1f}/5. "
        f"Academics {academic_avg:.1f}/5, Faculty {faculty_avg:.1f}/5, "
        f"Infrastructure {infra_avg:.1f}/5, Accommodation {accomm_avg:.1f}/5, "
        f"Social life {social_avg:.1f}/5. "
        f"Strongest areas: {', '.join(_top_aspects(placement_avg, academic_avg, infra_avg, faculty_avg))}."
    )
    chunks.append(_make_chunk(college_name, "placements", placements_text, row0))

    # Contact
    contacts = group[["Contact", "Email"]].drop_duplicates().head(3)
    contact_lines = [
        f"Phone: {r['Contact']}, Email: {r['Email']}"
        for _, r in contacts.iterrows()
        if pd.notna(r["Contact"]) and pd.notna(r["Email"])
    ]
    contact_text = (
        f"Contact information for {college_name}: "
        + ("; ".join(contact_lines) if contact_lines else "Visit the official website.")
    )
    chunks.append(_make_chunk(college_name, "contact", contact_text, row0))

    # Courses
    course_lines = [
        f"{stream}: {', '.join(sg['Preferred_Course'].unique().tolist())}"
        for stream, sg in group.groupby("Preferred_Stream")
    ]
    courses_text = (
        f"Courses and programmes at {college_name}: "
        + "; ".join(course_lines) + ". "
        f"Entrance exams accepted: {', '.join(exams)}."
    )
    chunks.append(_make_chunk(college_name, "courses", courses_text, row0))

    # Per-exam-per-course rank chunks (fine-grained for rank queries)
    for (exam, course), sg in group.groupby(["Entrance_Exam_Name", "Preferred_Course"]):
        ranks = sg["Entrance_Exam_Rank"].dropna()
        if ranks.empty:
            continue
        r_min, r_max = int(ranks.min()), int(ranks.max())
        pct   = sg["12th_Percentage"].dropna()
        ug_avg = sg["UG_Fee"].mean()
        
        if r_min == r_max:
            rank_str = f"Observed rank is {r_min} ({_rank_band(r_min)})."
        else:
            rank_str = f"Observed rank range {r_min} to {r_max} ({_rank_band(r_min)} to {_rank_band(r_max)})."

        rank_text = (
            f"Rank cutoff at {college_name} for {course} via {exam}: "
            f"{rank_str} "
            f"Minimum 12th percentage ~{pct.min():.1f}%, average {pct.mean():.1f}%. "
            f"Approximate UG fee Rs.{int(ug_avg):,} ({_fee_label(ug_avg)})."
        )
        cid = f"{college_name}_{exam}_{course}_rank".replace(" ", "_").lower()
        chunks.append({
            "id": cid,
            "text": rank_text,
            "metadata": {
                "college_name": college_name,
                "chunk_type": "cutoff",
                "exam": exam,
                "course": course,
                "rank_min": str(r_min),
                "rank_max": str(r_max),
                "state": str(row0.get("State", "")),
                "district": str(row0.get("District", "")),
            }
        })

    return chunks


def main():
    print("Loading college dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df)} rows, {df['College_Name'].nunique()} colleges found.")

    print("Loading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids, all_texts, all_metas = [], [], []
    for college_name, group in df.groupby("College_Name"):
        for chunk in build_chunks_for_group(college_name, group):
            all_ids.append(chunk["id"])
            all_texts.append(chunk["text"])
            all_metas.append(chunk["metadata"])

    print(f"Embedding {len(all_texts)} chunks ...")
    embeddings = model.encode(all_texts, show_progress_bar=True).tolist()

    print("Storing in ChromaDB ...")
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metas,
    )
    print(f"Ingest complete. {len(all_ids)} chunks stored in ChromaDB at '{CHROMA_PATH}'.")


if __name__ == "__main__":
    main()
