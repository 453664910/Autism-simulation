import sentence_transformers

from concordia.language_model import utils
import json
import os
from value_components.hardcoded_value_state import hardcoded_state_NT
from value_components.hardcoded_value_state import hardcoded_state_AS

# the root path of the project, this is used to import the D2A module
# 动态获取项目根目录（向上两级从 examples/D2A 到项目根目录）
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_current_file_dir))

# check point文件夹目录以及check point文件目录
checkpoint_folder = None
# checkpoint_folder = r'D:\Code\Autism-simulation\examples\D2A\result_folder\sim_result\2026-01-31_12-26-22\checkpoints\2026-01-31_12-36-51'
checkpoint_file = None
# checkpoint_file = r'D:\gitpro\Autism-simulation\examples\D2A\result_folder\sim_result\2026-01-09_15-00-13\checkpoints\2026-01-09_15-00-52\checkpoint_step_000000.pkl'

# how many episodes in each simulation, each episode is 20 minutes
episode_length = 60
# NUM_PLAYERS is the number of NT players in the simulation
NUM_PLAYERS = 10

# whether to use the language model, if set to True, No language model will be used
# use for debugging
disable_language_model = False

# the sentence transformer model used to encode the text
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

# whether to use the previous profile, if set to True, the previous profile will be used
# used to run with the same profile as the previous run
Use_Previous_profile = False
previous_profile_file = None
previous_profile = None
# TODO：修改previous_profile_file路径读取方式
# if Use_Previous_profile is True, the previous profile file should be provided
if Use_Previous_profile:
  previous_profile_file = os.path.join(r'examples\D2A\result_folder\indoor_result\Your folder name', 'Your previous profile name.json')
  try:
    with open(previous_profile_file, 'r') as f:
      previous_profile = json.load(f)
  except:
    raise ValueError('The previous profile file is not found.')
else:
  previous_profile = None

# the language model used to generate the text
# you can also use other models, detailed see the definition of language_model_setup
api_type = 'openai'
model_name = 'gpt-5.1-chat'
api_key='sk-iEI0qYvpEFmYOuX8jmTxSF4kCn9JbSFN4dCR73Mvd3VeAmZC'
device = 'cpu'
model = utils.language_model_setup(
    api_type=api_type,
    model_name=model_name,
    api_key=api_key,
    disable_language_model=disable_language_model,
)

# the desires that will be used in the simulation
wanted_desires = list(hardcoded_state_NT.keys())

###
# the hidden desires that will be used in the simulation
hidden_desires = []
###

# the path to store the result
current_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_name = 'result_folder'
current_folder_path = os.path.join(current_file_path, result_folder_name)
if not os.path.exists(current_folder_path):
  os.makedirs(current_folder_path)



