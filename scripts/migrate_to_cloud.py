"""
Script to migrate vector embeddings from Local Qdrant Storage to Qdrant Cloud.
"""
import os
import sys
from pathlib import Path

# Add backend directory to sys.path and change directory
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from qdrant_client import QdrantClient
from app.config import settings

def migrate():
    local_path = settings.QDRANT_PATH
    cloud_url  = settings.QDRANT_URL
    cloud_key  = settings.QDRANT_API_KEY

    if not cloud_url or not cloud_key:
        print("[ERROR] QDRANT_URL and QDRANT_API_KEY must be set in backend/.env")
        return

    print(f"[INFO] Connecting to Local Qdrant at: {local_path}")
    local_client = QdrantClient(path=str(local_path))

    print(f"[INFO] Connecting to Qdrant Cloud at: {cloud_url}")
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_key)

    collection_name = settings.QDRANT_COLLECTION

    # Scroll all points from local
    print(f"[INFO] Reading vectors from local collection '{collection_name}'...")
    offset = None
    all_points = []

    while True:
        records, next_offset = local_client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        all_points.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    print(f"[SUCCESS] Extracted {len(all_points)} vector points from local storage.")

    if not all_points:
        print("[INFO] No points to migrate.")
        return

    # Upsert to Cloud
    print(f"[INFO] Uploading {len(all_points)} points to Qdrant Cloud...")
    from qdrant_client.http.models import PointStruct

    cloud_points = [
        PointStruct(
            id=p.id,
            vector=p.vector,
            payload=p.payload,
        )
        for p in all_points
    ]

    cloud_client.upsert(
        collection_name=collection_name,
        points=cloud_points,
    )

    cloud_count = cloud_client.count(collection_name).count
    print(f"[COMPLETED] Migration Complete! Total points on Qdrant Cloud: {cloud_count}")

if __name__ == "__main__":
    migrate()
