import astervec
import os

opts = astervec.AsterVecDBOptions()
opts.dim = 128
db_dir = "./run/db/"
os.makedirs(db_dir, exist_ok=True)
if hasattr(opts, "vector_file_path"):
    opts.vector_file_path = os.path.join(db_dir, "vectors.bin")
if hasattr(opts, "log_file_path"):
    opts.log_file_path = os.path.join(db_dir, "astervec.log")
opts.reinit = True
db = astervec.AsterVecDB.open(db_dir, opts)

db.insert(1, [0.1] * 128)

db.close()
opts.reinit = False
db = astervec.AsterVecDB.open(db_dir, opts)

searchOpts = astervec.SearchOptions()
searchOpts.k = 10
results = db.search_knn([0.1] * 128, searchOpts)
print(results[0].id, results[0].distance)
