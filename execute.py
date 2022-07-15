import papermill as pm
import sys
import datetime
import signal


def write_to_file(s, frame_object=None, **kargs):
   """
   Signal handler - writes to file and don't stop the program
   Process still can be killed using: kill -9 <pid>
   """
   with open("out.txt", "a") as file:
      s = f"{str(datetime.datetime.now())}: {str(s)}\n"
      file.write(s)
      file.flush()


if __name__ == "__main__":
   """
   The script accepts notebook path as input (first argument). It executes the notebook with disabled
    signals to avoid OS interruption.
   A notebook with the execution outputs will be generated (second argument).
   The third argument is a dictionary with notebook parameters (see papermill for more information), optional.
   Especially useful for training NN.
   Note: since the notebook execution is triggered from this script, the PYTHONPATH is set to the 
   current directory. Therefore, imports may fail due to incorrect path. A simple solution is to located the notebook 
   in the same directory as the execute.py script.
   """
   catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
   for sig in catchable_sigs:
      signal.signal(sig, write_to_file)
   in_notebook, out_notebook = sys.argv[1], sys.argv[2]
   parameters = None
   if len(sys.argv) == 4:
      parameters = sys.argv[4]
   pm.execute_notebook(
      in_notebook,
      out_notebook,
      parameters
   )
