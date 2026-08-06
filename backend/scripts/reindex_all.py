"""Re-index every uploaded document after an index/graph schema upgrade."""
import argparse
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--yes", action="store_true", help="confirm destructive index rebuild")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Re-indexing replaces chunks, vectors, and graph provenance; pass --yes")

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        documents, skip, page_size = [], 0, 100
        while True:
            response = client.get(
                "/documents/", params={"skip": skip, "limit": page_size}
            )
            response.raise_for_status()
            page = response.json()["documents"]
            documents.extend(page)
            if len(page) < page_size:
                break
            skip += page_size
        for index, document in enumerate(documents, 1):
            document_id = document["id"]
            print(f"[{index}/{len(documents)}] re-indexing {document['original_name']}")
            client.post(f"/documents/{document_id}/reindex").raise_for_status()
            while True:
                status = client.get(f"/documents/{document_id}/status").raise_for_status().json()
                print(
                    f"  {status['progress_stage']}: "
                    f"{status['progress_current']}/{status['progress_total']}",
                    end="\r",
                )
                if status["status"] in {"completed", "failed"}:
                    print()
                    if status["status"] == "failed":
                        raise SystemExit(status.get("error_message") or f"Failed: {document_id}")
                    break
                time.sleep(2)


if __name__ == "__main__":
    main()
