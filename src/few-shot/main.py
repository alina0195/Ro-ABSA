from helper import format_data
from gpt import gpt
from zephyr import zephyr
from llama import llama

evaluation_df_absa = format_data("evaluation_data.json", "ABSA")
few_shot_absa = format_data("few_shot_data.json", "ABSA")

evaluation_df_atc = format_data("evaluation_data.json", "ATC")
few_shot_atc = format_data("few_shot_data.json", "ATC")

evaluation_df_alsc = format_data("evaluation_data.json", "ALSC")
few_shot_alsc = format_data("few_shot_data.json", "ALSC")

# #################################### GPT ABSA ####################################
# gpt(evaluation_df_absa, few_shot_absa, "ABSA")


# #################################### GPT ATC ####################################
# gpt(evaluation_df_atc, few_shot_atc, "ATC")


# #################################### GPT ALSC ####################################
# gpt(evaluation_df_alsc, few_shot_alsc, "ALSC")


# #################################### ZEPHYR ABSA ####################################
# zephyr(evaluation_df_absa, few_shot_absa, "ABSA")


# #################################### ZEPHYR ATC ####################################
# zephyr(evaluation_df_atc, few_shot_atc, "ATC")


# #################################### ZEPHYR ALSC ####################################
# zephyr(evaluation_df_alsc, few_shot_alsc, "ALSC")


# #################################### LLAMA ABSA ####################################
# llama(evaluation_df_absa, few_shot_absa, "ABSA")


# #################################### LLAMA ATC ####################################
# llama(evaluation_df_atc, few_shot_atc, "ATC")


# #################################### LLAMA ALSC ####################################
# llama(evaluation_df_alsc, few_shot_alsc, "ALSC")

