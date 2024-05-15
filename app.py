from datetime import datetime
import gradio as gr
import json, os
import requests
import numpy as np
from string import Template
import wave, io

# 在开头加入路径
import os, sys
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

import logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

from Synthesizers.base import Base_TTS_Synthesizer, Base_TTS_Task, get_wave_header_chunk
from src.common_config_manager import app_config, __version__

frontend_version = __version__

def load_character_emotions(character_name, characters_and_emotions):
    emotion_options = ["default"]
    emotion_options = characters_and_emotions.get(character_name, ["default"])

    return gr.Dropdown(emotion_options, value="default")

synthesizer_name = app_config.synthesizer

from importlib import import_module
from tools.i18n.i18n import I18nAuto

# 设置国际化支持
synthesizer_i18n_dir = f"Synthesizers/{synthesizer_name}/configs/i18n/locale"
character_manager_i18n_dir = os.path.join(os.path.dirname(__file__), "webuis/character_manager/i18n/locale")

i18n = I18nAuto(locale_paths=[synthesizer_i18n_dir, character_manager_i18n_dir])

# 动态导入合成器模块, 此处可写成 from Synthesizers.xxx import TTS_Synthesizer, TTS_Task
synthesizer_module = import_module(f"Synthesizers.{synthesizer_name}")
TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
TTS_Task = synthesizer_module.TTS_Task

# 创建合成器实例
tts_synthesizer:Base_TTS_Synthesizer = TTS_Synthesizer(debug_mode=True)

import soundfile as sf

all_gradio_components = {}

default_text = """鸟儿生来没有枷锁，那什么决定了我的命运？吹走雪白的花瓣，把我困于笼中。
这无尽的孤独，无法抹去我的幻想，总有一天我会做一个不受限制的梦。
让我的心勇敢展开翅膀，飞越那黑夜，寻找那皎白月光。
让云抚平我的刺痛，温柔地拭去我的忧伤。
Birds are born with no shackles, then what fatters my fate? Blown away the white petals, leave me trapped in the cage
The endless isolation, can’t wear down my illusion. Someday I’ll make a dream unchained
Let my heart bravely spread the wings, soaring past the night, to trace the bright moonlight
Let the clouds heal me of the stings, gently wipe the sorrow of my life"""

information = ""

try:
    with open("Information.md", "r", encoding="utf-8") as f:
        information = f.read()
except:
    pass
try:    
    max_text_length = app_config.max_text_length
except:
    max_text_length = -1

from webuis.builders.gradio_builder import GradioTabBuilder

ref_settings = tts_synthesizer.ui_config.get("ref_settings", [])
basic_settings = tts_synthesizer.ui_config.get("basic_settings", [])
advanced_settings = tts_synthesizer.ui_config.get("advanced_settings", [])
url_setting = tts_synthesizer.ui_config.get("url_settings", [])

tts_task_example: Base_TTS_Task = TTS_Task()
params_config = tts_task_example.params_config

has_character_param = True if "character" in params_config else False

global state

state = {
    'models_path': r"trained",
    'character_list': [],
    'edited_character_path': '',
    'edited_character_name': '',
    'ckpt_file_found': [],
    'pth_file_found': [],
    'wav_file_found': [],
}

global infer_config
infer_config = {
}

config_path = "gsv_config.json"
state["models_path"] = "trained"
locale_language = "auto"
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        state["models_path"] = config.get("models_path", "trained")

emotional_styles = [
    "default",
    "affectionate", "angry", "assistant", "calm", "chat", "cheerful", 
    "customerservice", "depressed", "disgruntled", "documentary-narration", "embarrassed", 
    "empathetic", "envious", "excited", "fearful", "friendly", "gentle", "hopeful", "lyrical", 
    "narration", "newscast", "poetry-reading", "sad", "serious", "shouting",
    "surprised", "whispering", "terrified", "unfriendly", "apologetic"
]

language_list = ["auto", "zh", "en", "ja", "all_zh", "all_ja"]
translated_language_dict = {}
for language in language_list:
    translated_language_dict[language] = language
    translated_language_dict[i18n(language)] = language
    translated_language_dict["多语种混合"] = "auto"

all_emotion_num = len(emotional_styles)

from time import time as ttime

def get_audio(*data, streaming=False):
    data = dict(zip([key for key in all_gradio_components.keys()], data))
    data["stream"] = streaming
    
    if data.get("text") in ["", None]:
        gr.Warning(i18n("文本不能为空"))
        return None, None
    try:
        task: Base_TTS_Task = tts_synthesizer.params_parser(data)
        t2 = ttime()
        
        if not streaming:
            # if synthesizer_name == "remote":
            #     save_path = tts_synthesizer.generate(task, return_type="filepath")
            #     yield save_path
            # else:
            #     gen = tts_synthesizer.generate(task, return_type="numpy")
            #     yield next(gen)
            if synthesizer_name == "remote":
                save_path = tts_synthesizer.generate(task, return_type="filepath")
                yield save_path
                with open(save_path, 'rb') as audio_file:
                    result_audio = audio_file.read()
                save_infer_record(result_audio, data)
            else:
                gen = tts_synthesizer.generate(task, return_type="numpy")
                result_audio = next(gen)
                # print("result audio:", result_audio)
                sample_rate, audio = result_audio
                audio = audio.astype(np.int16).tobytes()  # 将 numpy 数组转换为 int16 类型的 bytes
                yield result_audio
                save_infer_record(audio, data, sample_rate)
        else:
            gen = tts_synthesizer.generate(task, return_type="numpy")
            sample_rate = 32000 if task.sample_rate in [None, 0] else task.sample_rate
            yield get_wave_header_chunk(sample_rate=sample_rate)
            for chunk in gen:
                yield chunk
        
    except Exception as e:
        gr.Warning(f"Error: {e}")

from functools import partial
get_streaming_audio = partial(get_audio, streaming=True)

def stopAudioPlay():
    return

global characters_and_emotions_dict
characters_and_emotions_dict = {}

def change_trained_dir(trained_dir="", character="", emotion="default"):
    trained_dir_names = [os.path.basename(f.path) for f in os.scandir('.') if f.is_dir() and os.path.basename(f.path).startswith('trained')]
    if trained_dir in trained_dir_names:
        trained_dir_name = trained_dir
    else:
        trained_dir_name = trained_dir_names[0]
    os.environ['models_path'] = trained_dir_name

    return (
        gr.Dropdown(trained_dir_names, value=trained_dir_name, label=i18n("模型文件夹路径")),
        *change_character_list(character, emotion)
    )

def change_character_list(character="", emotion="default"):
    characters_and_emotions = {}
    print(f"trained模型地址: {os.environ.get('models_path', 'trained')}")
    
    try:
        tts_synthesizer.ui_config['models_path'] = os.environ.get('models_path', 'trained')
        characters_and_emotions = tts_synthesizer.get_characters()
        character_names = [i for i in characters_and_emotions]
        if len(character_names) != 0:
            if character in character_names:
                character_name_value = character
            else:
                character_name_value = character_names[0]
        else:
            character_name_value = ""
        emotions = characters_and_emotions.get(character_name_value, ["default"])
        emotion_value = emotion

    except:
        character_names = []
        character_name_value = ""
        emotions = ["default"]
        emotion_value = "default"
        characters_and_emotions = {}

    return (
        gr.Dropdown(character_names, value=character_name_value, label=i18n("选择角色")),
        gr.Dropdown(emotions, value=emotion_value, label=i18n("情感列表"), interactive=True),
        characters_and_emotions,
    )

def cut_sentence_multilang(text, max_length=30):
    if max_length == -1:
        return text, ""
    word_count = 0
    in_word = False
    
    for index, char in enumerate(text):
        if char.isspace():
            in_word = False
        elif char.isascii() and not in_word:
            word_count += 1
            in_word = True
        elif not char.isascii():
            word_count += 1
        if word_count > max_length:
            return text[:index], text[index:]
    
    return text, ""

def run_inference_tab():
    with gr.Blocks() as inference_app:
        gr.Markdown(information)
        with gr.Row():
            max_text_length_tip = "" if max_text_length == -1 else f"( "+i18n("最大允许长度")+ f" : {max_text_length} ) "
            text = gr.Textbox(
                value=default_text, label=i18n("输入文本")+max_text_length_tip, interactive=True, lines=8
            )
            text.blur(lambda x: gr.update(value=cut_sentence_multilang(x,max_length=max_text_length)[0]), [text], [text])
            all_gradio_components["text"] = text
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab(label=i18n("角色选项"), visible=has_character_param):
                        with gr.Group():
                            (
                                trained_dir,
                                character,
                                emotion,
                                characters_and_emotions_,
                            ) = change_trained_dir()
                            characters_and_emotions = gr.State(characters_and_emotions_)
                            scan_character_list = gr.Button(
                                i18n("扫描人物列表"), variant="secondary"
                            )
                        all_gradio_components["trained_dir"] = trained_dir
                        all_gradio_components["character"] = character
                        all_gradio_components["emotion"] = emotion
                        trained_dir.change()

                        character.change(
                            load_character_emotions,
                            inputs=[character, characters_and_emotions],
                            outputs=[emotion],
                        )

                        scan_character_list.click(
                            change_trained_dir,
                            inputs=[trained_dir, character, emotion],
                            outputs=[
                                trained_dir,
                                character,
                                emotion,
                                characters_and_emotions,
                            ],
                        )
                    if len(ref_settings) > 0:
                        with gr.Tab(label=i18n("参考设置")):
                            ref_settings_tab = GradioTabBuilder(
                                ref_settings, params_config
                            )
                            ref_settings_components = ref_settings_tab.build()
                            all_gradio_components.update(ref_settings_components)
            with gr.Column(scale=2):
                with gr.Tabs():
                    if len(basic_settings) > 0:
                        with gr.Tab(label=i18n("基础选项")):
                            basic_settings_tab = GradioTabBuilder(
                                basic_settings, params_config
                            )
                            basic_settings_components = basic_settings_tab.build()
                            all_gradio_components.update(basic_settings_components)
            with gr.Column(scale=2):
                with gr.Tabs():
                    if len(advanced_settings) > 0:
                        with gr.Tab(label=i18n("高级选项")):
                            advanced_settings_tab = GradioTabBuilder(
                                advanced_settings, params_config
                            )
                            advanced_settings_components = advanced_settings_tab.build()
                            all_gradio_components.update(advanced_settings_components)
                    if len(url_setting) > 0:
                        with gr.Tab(label=i18n("URL设置")):
                            url_setting_tab = GradioTabBuilder(url_setting, params_config)
                            url_setting_components = url_setting_tab.build()
                            all_gradio_components.update(url_setting_components)
        with gr.Tabs():
            with gr.Tab(label=i18n("请求完整音频")):
                with gr.Row():
                    get_full_audio_button = gr.Button(i18n("生成音频"), variant="primary")
                    full_audio = gr.Audio(
                        None, label=i18n("音频输出"), type="filepath", streaming=False
                    )
                    get_full_audio_button.click(lambda: gr.update(interactive=False), None, [get_full_audio_button]).then(
                        get_audio,
                        inputs=[value for key, value in all_gradio_components.items()],
                        outputs=[full_audio],
                    ).then(lambda: gr.update(interactive=True), None, [get_full_audio_button])
            with gr.Tab(label=i18n("流式音频")):
                with gr.Row():
                    get_streaming_audio_button = gr.Button(i18n("生成流式音频"), variant="primary")
                    streaming_audio = gr.Audio(
                        None, label=i18n("音频输出"), type="filepath", streaming=True, autoplay=True
                    )
                    get_streaming_audio_button.click(lambda: gr.update(interactive=False), None, [get_streaming_audio_button]).then(
                        get_streaming_audio,
                        inputs=[value for key, value in all_gradio_components.items()],
                        outputs=[streaming_audio],
                    ).then(lambda: gr.update(interactive=True), None, [get_streaming_audio_button])

        gr.HTML("<hr style='border-top: 1px solid #ccc; margin: 20px 0;' />")
        gr.HTML(
            f"""<p>{i18n("这是GSVI。")}{i18n("，当前版本：")}<a href="https://www.yuque.com/xter/zibxlp/awo29n8m6e6soru9">{frontend_version}</a>  {i18n("项目开源地址：")} <a href="https://github.com/X-T-E-R/GPT-SoVITS-Inference">Github</a></p>
                <p>{i18n("若有疑问或需要进一步了解，可参考文档：")}<a href="{i18n("https://www.yuque.com/xter/zibxlp")}">{i18n("点击查看详细文档")}</a>。</p>"""
        )
    return inference_app

def generate_info_bar():
    current_character_textbox = gr.Textbox(value=state['edited_character_name'], label=i18n("当前人物"), interactive=False)
    version_textbox = gr.Textbox(value=infer_config['version'], label=i18n("版本"), interactive=True)
    gpt_model_dropdown = gr.Dropdown(choices=state['ckpt_file_found'], label=i18n("GPT模型路径"), interactive=True, value=infer_config['gpt_path'], allow_custom_value=True)
    sovits_model_dropdown = gr.Dropdown(choices=state['pth_file_found'], label=i18n("Sovits模型路径"), interactive=True, value=infer_config['sovits_path'], allow_custom_value=True)
    column_items = [current_character_textbox, version_textbox, gpt_model_dropdown, sovits_model_dropdown]
    index = 0
    for item in infer_config['emotion_list']:
        emotion, details = item
        index += 1
        column_items.append(gr.Number(index, visible=True, scale=1))
        column_items.append(gr.Dropdown(choices=[(i18n(language), language) for language in language_list], value=translated_language_dict[details['prompt_language']], visible=True, interactive=True, scale=3, label=i18n("提示语言")))
        column_items.append(gr.Dropdown(choices=emotional_styles, value=emotion, visible=True, interactive=True, scale=3, allow_custom_value=True, label=i18n("情感风格")))
        column_items.append(gr.Dropdown(choices=state["wav_file_found"], visible=True, value=details['ref_wav_path'], scale=8, allow_custom_value=True, label=i18n("参考音频路径")))
        column_items.append(gr.Textbox(value=details['prompt_text'], visible=True, scale=8, interactive=True, label=i18n("提示文本")))
        column_items.append(gr.Audio(os.path.join(state["edited_character_path"], details['ref_wav_path']), visible=True, scale=8, label=i18n("音频预览")))

    for i in range(all_emotion_num - index):
        column_items.append(gr.Number(i, visible=False))
        column_items.append(gr.Dropdown(visible=False))
        column_items.append(gr.Dropdown(visible=False))
        column_items.append(gr.Dropdown(visible=False))
        column_items.append(gr.Textbox(visible=False))
        column_items.append(gr.Audio(None, visible=False))

    return column_items

def load_json_to_state(data):
    infer_config['version'] = data.get('version','')
    emotional_list = data.get('emotion_list',{})
    for emotion, details in emotional_list.items():
        infer_config['emotion_list'].append([emotion,details])
    infer_config['gpt_path'] = data['gpt_path']
    infer_config['sovits_path'] = data['sovits_path']
    return generate_info_bar()

def split_file_name(file_name):
    try:
        base_name = os.path.basename(file_name)
    except:
        base_name = file_name
  
    final_name = os.path.splitext(base_name)[0]
    pattern = r"【.*?】"
    
    return re.sub(pattern, "", final_name).replace('说话-', '')

def clear_infer_config():
    global infer_config
    infer_config = {
        'version': '1.0.1',
        'gpt_path': '',
        'sovits_path': '',
        'emotion_list': [],
    }

clear_infer_config()

def read_json_from_file(character_dropdown, models_path):
    state['edited_character_name'] = character_dropdown  
    state['models_path'] = models_path 
    state['edited_character_path'] = os.path.join(state['models_path'], state['edited_character_name'])
    state['ckpt_file_found'], state['pth_file_found'], state['wav_file_found'] = scan_files(state['edited_character_path'])
    gr.Info(i18n(f"当前人物变更为: {state['edited_character_name']}"))
    print(i18n(f"当前人物变更为: {state['edited_character_name']}"))
    clear_infer_config()
    json_path = os.path.join(state['edited_character_path'], "infer_config.json")
    if not os.path.exists(json_path):
        auto_generate_json(character_dropdown, models_path)
        save_json()
    
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        if 'ref_wav_path' not in data['emotion_list']['default']:
            auto_generate_json(character_dropdown, models_path)
            save_json()
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        return load_json_to_state(data)

def auto_generate_all_json(models_path):
    for subfolder in state['character_list']:
        read_json_from_file(subfolder, models_path)

def save_json():
    if infer_config['gpt_path'] == '' or infer_config['gpt_path'] is None:
        gr.Error(i18n("缺失某些项，保存失败！"))
        raise Exception(i18n("缺失某些项，保存失败！"))
    json_path = os.path.join(state['edited_character_path'], "infer_config.json")
    data = {
        'version': infer_config['version'],
        'gpt_path': infer_config['gpt_path'],
        'sovits_path': infer_config['sovits_path'],
        i18n("简介"): i18n(r"这是一个配置文件适用于https://github.com/X-T-E-R/TTS-for-GPT-soVITS，是一个简单好用的前后端项目"),
        'emotion_list': {}
    }
    for item in infer_config['emotion_list']:
        data['emotion_list'][item[0]] = item[1]
    try:
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        gr.Info(i18n("保存成功！"))
    except:
        gr.Error(i18n("文件打开失败，保存失败！"))
        raise Exception(i18n("保存失败！"))

def scan_files(character_path):
    ckpt_file_found = []
    pth_file_found = []
    wav_file_found = []
    for dirpath, dirnames, filenames in os.walk(character_path):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            rev_path = os.path.relpath(full_path, character_path)
            print(full_path)
            if file.lower().endswith(".ckpt"):
                ckpt_file_found.append(rev_path)
            elif file.lower().endswith(".pth"):
                pth_file_found.append(rev_path)
            elif file.lower().endswith(".wav"):
                wav_file_found.append(rev_path)
            elif file.lower().endswith(".mp3"):
                wav_file_found.append(rev_path)
    return ckpt_file_found, pth_file_found, wav_file_found

def auto_generate_json(character_dropdown, models_path):
    state['edited_character_name'] = character_dropdown  
    state['models_path'] = models_path 
    state['edited_character_path'] = os.path.join(state['models_path'], state['edited_character_name'])
    
    print(i18n(f"当前人物变更为: {state['edited_character_name']}"))
    clear_infer_config()
    character_path = state['edited_character_path']
    
    ckpt_file_found, pth_file_found, wav_file_found = scan_files(character_path)
   
    if len(ckpt_file_found) == 0 or len(pth_file_found) == 0:
        gr.Error(i18n("找不到模型文件！请把有效文件放置在文件夹下！！！"))
        raise Exception(i18n("找不到模型文件！请把有效文件放置在文件夹下！！！"))
    else:
        state['ckpt_file_found'] = ckpt_file_found
        state['pth_file_found'] = pth_file_found
        state['wav_file_found'] = wav_file_found
        gpt_path = ckpt_file_found[0]
        sovits_path = pth_file_found[0]

        infer_config['gpt_path'] = gpt_path
        infer_config['sovits_path'] = sovits_path
    
    if len(wav_file_found) == 0:
        return generate_info_bar()
    else:
        return add_emotion()

def scan_subfolder(models_path):
    subfolders = [os.path.basename(f.path) for f in os.scandir(models_path) if f.is_dir()]
    state['models_path'] = models_path
    state['character_list'] = subfolders
    print(i18n("扫描模型文件夹:")+models_path)
    print(i18n(f"找到的角色列表:") + str(subfolders))
    gr.Info(i18n(f"找到的角色列表:") + str(subfolders))
    d2 = gr.Dropdown(subfolders)
    return d2

def add_emotion():
    unused_emotional_style = ''
    for style in emotional_styles:
        style_in_list = False
        for item in infer_config['emotion_list']:
            if style == item[0]:
                style_in_list = True
                break
        if not style_in_list:
            unused_emotional_style = style
            break
    
    ref_wav_path = state['wav_file_found'][0]
    infer_config['emotion_list'].append([f'{unused_emotional_style}', {
        'ref_wav_path': ref_wav_path, 'prompt_text': split_file_name(ref_wav_path), 'prompt_language': 'auto'}])
    return generate_info_bar()

def change_pt_files(version_textbox, sovits_model_dropdown, gpt_model_dropdown):
    infer_config['version'] = version_textbox
    infer_config['sovits_path'] = sovits_model_dropdown
    infer_config['gpt_path'] = gpt_model_dropdown
    pass

def change_parameters(index, wav_path, emotion_list, prompt_language, prompt_text=""):
    index = int(index)
    
    if prompt_text == "" or prompt_text is None:
        prompt_text = split_file_name(wav_path)
    
    infer_config['emotion_list'][index-1][0] = emotion_list
    infer_config['emotion_list'][index-1][1]['ref_wav_path'] = wav_path
    infer_config['emotion_list'][index-1][1]['prompt_text'] = prompt_text
    infer_config['emotion_list'][index-1][1]['prompt_language'] = prompt_language

    return gr.Dropdown(value=wav_path), gr.Dropdown(value=emotion_list), gr.Dropdown(value=prompt_language), gr.Textbox(value=prompt_text), gr.Audio(os.path.join(state["edited_character_path"], wav_path))

def run_model_tab():
    with gr.Blocks() as model_app:
        with gr.Row() as status_bar:
            trained_dir_names = [os.path.basename(f.path) for f in os.scandir('.') if f.is_dir() and os.path.basename(f.path).startswith('trained')]
            trained_dir_name = trained_dir_names[0]
            # models_path = gr.Textbox(value=state["models_path"], label=i18n("模型文件夹路径"), scale=3)
            models_path = gr.Dropdown(trained_dir_names, value=trained_dir_name, label=i18n("模型文件夹路径"), scale=3)
            scan_button = gr.Button(i18n("扫描"), scale=1, variant="primary")
            character_dropdown = gr.Dropdown([], label=i18n("选择角色"), scale=3)
            read_info_from_json_button = gr.Button(i18n("从json中读取（无则生成）"), size="lg", scale=2, variant="secondary")
            auto_generate_info_button = gr.Button(i18n("生成所有info（有则跳过）"), size="lg", scale=2, variant="primary")
            
            scan_button.click(scan_subfolder, inputs=[models_path], outputs=[character_dropdown])
        
        gr.HTML(i18n("""<p>这是模型管理界面，为了实现对多段参考音频分配情感设计，如果您只有一段可不使用这个界面</p><p>若有疑问或需要进一步了解，可参考文档：<a href="https://www.yuque.com/xter/zibxlp/hme8bw2r28vad3le">点击查看详细文档</a>。</p>"""))
        gr.Markdown(i18n("请修改后点击下方按钮进行保存"))

        with gr.Row() as submit_bar:
            save_json_button = gr.Button(i18n("保存json\n（可能不会有完成提示，没报错就是成功）"), scale=2, variant="primary")
            save_json_button.click(save_json)
        
        with gr.Row():
            with gr.Column(scale=1):
                current_character_textbox = gr.Textbox(value=state['edited_character_name'], label=i18n("当前人物"), interactive=False)
                version_textbox = gr.Textbox(value=infer_config['version'], label=i18n("版本"))
                gpt_model_dropdown = gr.Dropdown(choices=state['ckpt_file_found'], label=i18n("GPT模型路径"))
                sovits_model_dropdown = gr.Dropdown(choices=state['pth_file_found'], label=i18n("Sovits模型路径"))
                version_textbox.blur(change_pt_files, inputs=[version_textbox, sovits_model_dropdown, gpt_model_dropdown], outputs=None)
                gpt_model_dropdown.input(change_pt_files, inputs=[version_textbox, sovits_model_dropdown, gpt_model_dropdown], outputs=None)
                sovits_model_dropdown.input(change_pt_files, inputs=[version_textbox, sovits_model_dropdown, gpt_model_dropdown], outputs=None)
                column_items = [current_character_textbox, version_textbox, gpt_model_dropdown, sovits_model_dropdown]
            
            with gr.Column(scale=3):
                add_emotion_button = gr.Button(i18n("添加情感"), size="lg", scale=2, variant="primary")

                for index in range(all_emotion_num):
                    with gr.Row() as emotion_row:
                        row_index = gr.Number(visible=False)
                        emotional_list = gr.Dropdown(visible=False)
                        prompt_language = gr.Dropdown(visible=False)
                        wav_path = gr.Dropdown(visible=False)
                        prompt_text = gr.Textbox(visible=False)
                        audio_preview = gr.Audio(visible=False, type="filepath")

                        emotional_list.input(change_parameters, inputs=[row_index, wav_path, emotional_list, prompt_language, prompt_text], outputs=[wav_path, emotional_list, prompt_language, prompt_text, audio_preview])
                        prompt_language.input(change_parameters, inputs=[row_index, wav_path, emotional_list, prompt_language, prompt_text], outputs=[wav_path, emotional_list, prompt_language, prompt_text, audio_preview])
                        wav_path.input(change_parameters, inputs=[row_index, wav_path, emotional_list, prompt_language], outputs=[wav_path, emotional_list, prompt_language, prompt_text, audio_preview])
                        prompt_text.input(change_parameters, inputs=[row_index, wav_path, emotional_list, prompt_language, prompt_text], outputs=[wav_path, emotional_list, prompt_language, prompt_text, audio_preview])
                        
                        column_items.append(row_index)
                        column_items.append(prompt_language)
                        column_items.append(emotional_list)
                        column_items.append(wav_path)
                        column_items.append(prompt_text)
                        column_items.append(audio_preview)

        add_emotion_button.click(add_emotion, outputs=column_items)
        read_info_from_json_button.click(read_json_from_file, inputs=[character_dropdown, models_path], outputs=column_items)
        auto_generate_info_button.click(auto_generate_all_json, inputs=[models_path])
    
    return model_app


import os
import json
import socket
from datetime import datetime

# 创建history文件夹
history_dir = "history"
if not os.path.exists(history_dir):
    os.makedirs(history_dir)

def get_client_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def save_infer_record(result_audio, config, sample_rate=32000):
    user_ip = get_client_ip()
    infer_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_folder = config.get("trained_dir", "unknown")
    character = config.get("character", "unknown")
    emotion = config.get("emotion", "default")
    text = config.get("text", "")
    text_snippet = text[:20].replace("__", " ")

    filename = f"{user_ip}__{infer_time}__{model_folder}__{character}__{emotion}__{text_snippet}.json"
    file_path = os.path.join(history_dir, filename)

    audio_path = file_path.replace(".json", ".wav")
    with wave.open(audio_path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(result_audio)

    infer_record = {
        "user_ip": user_ip,
        "time": infer_time,
        "config": config
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(infer_record, f, ensure_ascii=False, indent=4)

def load_infer_records():
    records = []
    for filename in os.listdir(history_dir):
        if filename.endswith(".json"):
            parts = filename.split("__")
            if len(parts) == 6:
                user_ip, infer_time, model_folder, character, emotion, text_snippet = parts
                records.append({
                    "user_ip": user_ip,
                    "time": infer_time,
                    "model_folder": model_folder,
                    "character": character,
                    "emotion": emotion,
                    "text_snippet": text_snippet.replace(".json", "")
                })
    return records

def display_record_detail(record):
    file_path = os.path.join(history_dir, f"{record['user_ip']}__{record['time']}__{record['model_folder']}__{record['character']}__{record['emotion']}__{record['text_snippet']}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        record_detail = json.load(f)
    audio_path = file_path.replace(".json", ".wav")
    return record_detail["config"], audio_path

def filter_records(records, user_ip, time, model_folder, character, emotion, text):
    filtered = []
    for record in records:
        if user_ip and user_ip not in record["user_ip"]:
            continue
        if time and time not in record["time"]:
            continue
        if model_folder and model_folder not in record["config"]["trained_dir"]:
            continue
        if character and character not in record["config"]["character"]:
            continue
        if emotion and emotion not in record["config"]["emotion"]:
            continue
        if text and text not in record["config"]["text"]:
            continue
        filtered.append(record)
    return filtered

def filter_and_display_records(user_ip, time, model_folder, character, emotion, text):
    records = load_infer_records()
    filtered_records = filter_records(records, user_ip, time, model_folder, character, emotion, text)
    # 将字典转换为列表
    filtered_records_list = [[record["user_ip"], record["time"], record["model_folder"], record["character"], record["emotion"], record["text_snippet"]] for record in filtered_records]
    return filtered_records_list


def run_records_tab():
    with gr.Blocks() as records_app:
        with gr.Row():
            user_ip = gr.Textbox(label="推理用户（IP地址）")
            time = gr.Textbox(label="推理时间")
            model_folder = gr.Textbox(label="模型文件夹")
            character = gr.Textbox(label="人物角色")
            emotion = gr.Textbox(label="情感")
            text = gr.Textbox(label="输入文本")
            search_button = gr.Button("搜索")

        records_output = gr.Dataframe(headers=["推理用户", "推理时间", "模型文件夹", "人物角色", "情感", "文本片段"], datatype=["str", "str", "str", "str", "str", "str"])
        record_detail_output = gr.JSON(label="推理详情")
        audio_output = gr.Audio(label="生成音频", type="filepath")

        def filter_and_display_records(user_ip, time, model_folder, character, emotion, text):
            records = load_infer_records()
            filtered_records = filter_records(records, user_ip, time, model_folder, character, emotion, text)
            # 将字典转换为列表
            filtered_records_list = [[record["user_ip"], record["time"], record["model_folder"], record["character"], record["emotion"], record["text_snippet"]] for record in filtered_records]
            return filtered_records_list

        def load_record_detail(record_index):
            records = load_infer_records()
            selected_record = records[record_index]
            record_config, audio_path = display_record_detail(selected_record)
            return record_config, audio_path

        def on_select_record(evt: gr.SelectData):
            record_index = evt.index[0]  # 只取第一个索引
            record_config, audio_path = load_record_detail(record_index)
            return record_config, audio_path

        search_button.click(
            filter_and_display_records,
            inputs=[user_ip, time, model_folder, character, emotion, text],
            outputs=[records_output]
        )

        records_output.select(
            on_select_record,
            outputs=[record_detail_output, audio_output]
        )

        records_app.load(lambda: filter_and_display_records("", "", "", "", "", ""), outputs=[records_output])
    return records_app


if __name__ == '__main__':
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.Tab(label="推理"):
                run_inference_tab()
            with gr.Tab(label="模型配置"):
                run_model_tab()
            with gr.Tab(label="生成记录"):
                run_records_tab()
    app.launch(server_port=9872, inbrowser=True, share=False)
