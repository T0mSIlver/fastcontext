import json
import subprocess


# docker run with a given image and run a ls command in the given workspace, and return the output
def run_docker_command(image: str, workspace: str, timeout: int = 30) -> None | str:
    # cmd_str = f"docker run --rm {image} bash -c \"cd {workspace} && ls\""
    cmd = ["docker", "run", "--rm", image, "bash", "-c", f"cd {workspace} && ls"]
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n = 0
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
    with open(output_file, "w") as f:
        for s in samples:
            ls_output = run_docker_command(s["image_name"], s["workspace"])
            if ls_output is None:
                continue
            s["workspace_ls"] = ls_output
            f.write(json.dumps(s) + "\n")
            n += 1
    print(f"Processed {n} / {len(samples)} samples")
