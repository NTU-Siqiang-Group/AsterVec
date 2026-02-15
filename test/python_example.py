import lsm_vec
import os

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
db_dir = "./run/db/"
if hasattr(opts, "vector_file_path"):
    opts.vector_file_path = os.path.join(db_dir, "vectors.bin")
if hasattr(opts, "log_file_path"):
    opts.log_file_path = os.path.join(db_dir, "lsmvec.log")
opts.reinit = True
db = lsm_vec.LSMVecDB.open(db_dir, opts)

db.insert(1, [0.1] * 128)

searchOpts = lsm_vec.SearchOptions()
searchOpts.k = 10
results = db.search_knn([0.1] * 128, searchOpts)
print(results[0].id, results[0].distance)
