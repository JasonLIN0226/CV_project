from pathlib import Path
import argparse, datetime, hashlib, json, os, socket, subprocess

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def run(argv):
    p=subprocess.run(argv,capture_output=True,text=True)
    return {"argv":argv,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
def main():
    a=argparse.ArgumentParser();a.add_argument("--job",required=True,type=int);q=a.parse_args()
    root=Path("/scratch/xl4358/phase1_space_postartifact_handoff")
    bench=Path("/scratch/xl4358/phase1_space_paper_benchmark")
    out=Path("/scratch/xl4358/phase1_space_artifact_results/job_7330")
    dest=out/f"root_recovery_boundary_{q.job}.json"
    if dest.exists():raise RuntimeError("existing recovery boundary, no blind repeat")
    current=run(["/opt/slurm/bin/scontrol","show","job","-o",str(q.job)])
    fields=dict(x.split("=",1) for x in current["stdout"].split() if "=" in x)
    if current["returncode"] or fields.get("Account")!="csci_ga_2271-2026fa" or fields.get("JobState")!="RUNNING" or "gres/gpu" in fields.get("AllocTRES",""):raise RuntimeError("actual running zeroGPU course allocation required")
    if fields.get("NodeList")!=socket.gethostname().split(".")[0]:raise RuntimeError("actual host allocation mismatch")
    if fields.get("TimeLimit")!="00:15:00":raise RuntimeError("fixed fifteen minute metadata allocation required")
    tres=dict(x.split("=",1) for x in fields["AllocTRES"].split(",") if "=" in x)
    if int(tres.get("cpu","0"))<4 or tres.get("mem") not in ("16G","16384M"):raise RuntimeError("metadata resource boundary differs")
    active=run(["/opt/slurm/bin/squeue","-u","xl4358","-h","-o","%i|%j|%T"])
    if active["returncode"] or any(r.split("|")[0]!=str(q.job) for r in active["stdout"].splitlines()):raise RuntimeError("unexpected concurrent allocation")
    sacct=run(["/opt/slurm/bin/sacct","-j","7313,7330,7331","-n","-P","--format=JobIDRaw,JobName%80,State%80,ExitCode,Account,Partition,Start,End,Elapsed,AllocTRES%120,NodeList%80,Reason%100"])
    rows=[r.split("|") for r in sacct["stdout"].splitlines()]
    for parent in (7313,7330):
        if not any(r[0]==str(parent) and r[2:5]==["COMPLETED","0:0","csci_ga_2271-2026fa"] for r in rows):raise RuntimeError("original complete science parent not proved")
    cancelled=[r for r in rows if r[0]=="7331"]
    if len(cancelled)!=1 or not cancelled[0][2].startswith("CANCELLED"):raise RuntimeError("ended metadata cancellation boundary differs")
    if any(r[0].startswith("7331.") for r in rows):raise RuntimeError("metadata batch/step exists, inspect outcome before recovery")
    forbidden=["benchmark_protocol.json","source_lock.json","submission_attempt.json","benchmark_submission.json"]
    if any((bench/n).exists() for n in forbidden) or (root/"completed_artifact_numeric_proof.json").exists():raise RuntimeError("unknown prior binding or proof outcome; stop without retry")
    logs=sorted(str(f) for f in (root/"logs").iterdir())
    if logs:raise RuntimeError("old metadata has logs, inspect before recovery")
    locks=[]
    for folder,name,expected in [(root,"source_lock.json","d1d1e6984abd0fa0c8c31505d5f6bdae011d7fbde6ddc3f1a4a95f4de550e014"),(bench,"software_lock.json","caf632d26d7d2c4f7598621433f92de3af53e30ab4473dc33f7782b6d2ec63cc")]:
        lock=folder/name
        if sha(lock)!=expected:raise RuntimeError("frozen lock hash differs")
        info=json.loads(lock.read_text())
        for name,value in info["files"].items():
            f=folder/name
            if f.is_symlink() or sha(f)!=value:raise RuntimeError("frozen source changed")
        locks.append({"root":str(folder),"lock_sha256":sha(lock),"files_verified":len(info["files"])})
    report=json.loads((out/"artifact_report.json").read_text())
    output=[]
    for r in report["outputs"]:
        f=Path(r["path"])
        if f.is_symlink() or f.stat().st_size!=r["bytes"] or sha(f)!=r["sha256"]:raise RuntimeError("complete artifact output boundary changed")
        output.append({"path":str(f),"bytes":f.stat().st_size,"sha256":sha(f)})
    record={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"actual_recovery_allocation":q.job,"current_allocation":current,"squeue":active,"original_sacct":sacct,"original_metadata_scontrol":run(["/opt/slurm/bin/scontrol","show","job","-o","7331"]),"failed_node_current_scontrol":run(["/opt/slurm/bin/scontrol","show","node","-o","b-10-144"]),"no_original_metadata_batch_or_logs":True,"no_unknown_benchmark_binding_or_attempt":True,"source_locks":locks,"current_artifact_outputs":output,"artifact_report_sha256":sha(out/"artifact_report.json"),"frozen_controller_numeric_model_source_label_proof_still_required_before_registration":True,"no_training_or_artifact_repeated":True}
    dest.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"boundary":str(dest),"sha256":sha(dest),"actual_job":q.job,"ready_for_once_frozen_controller":True}))
if __name__=="__main__":main()
