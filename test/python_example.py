import lsm_vec
import os

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
db_dir = "./run/db/"
os.makedirs(db_dir, exist_ok=True)
if hasattr(opts, "vector_file_path"):
    opts.vector_file_path = os.path.join(db_dir, "vectors.bin")
if hasattr(opts, "log_file_path"):
    opts.log_file_path = os.path.join(db_dir, "lsmvec.log")
opts.reinit = True
db = lsm_vec.LSMVecDB.open(db_dir, opts)

db.insert(1, [0.1] * 128)

db.close()
opts.reinit = False
db = lsm_vec.LSMVecDB.open(db_dir, opts)

searchOpts = lsm_vec.SearchOptions()
searchOpts.k = 10
results = db.search_knn([0.1] * 128, searchOpts)
print(results[0].id, results[0].distance)
