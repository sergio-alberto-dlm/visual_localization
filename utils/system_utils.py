
def default_num_workers() -> int:
    try:
        import multiprocessing as mp 
        return max(1, mp.cpu_count() // 2)
    except Exception:
        return 4

def try_import_faiss():
    try:
        import faiss 
        return faiss 
    except Exception:
        return None 

