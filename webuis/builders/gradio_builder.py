
import gradio as gr

from Synthesizers.base import ParamItem
from typing import List, Dict, Literal, Optional, Any, Union

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(locale_path="Synthesizers/gsv_fast/configs/i18n/locale")

class GradioTabBuilder:
    """
    A class for building Gradio tabs.

    Attributes:
        component_group_list (List[List[ParamItem]]): A list of component groups.
        gradio_input_dict (Dict[str, Any]): A dictionary of Gradio inputs.

    Methods:
        __init__(self, component_name_list: List[Union[str, List[str]]], params_config: Dict[str, ParamItem]): Initializes the GradioTabBuilder object.
        add_input(self, name: str, input: Any): Adds an input to the gradio_input_dict.
        build_group(self, component_list: List[ParamItem]): Builds a group of components.
        build(self): Builds the Gradio tab and returns the gradio_input_dict.
    """

    component_group_list: List[List[ParamItem]]
    gradio_input_dict: Dict[str, Any]
    def __init__(self, component_name_list: List[Union[str, List[str]]], params_config: Dict[str, ParamItem]):
        """
        Initializes the GradioTabBuilder object.

        Args:
            component_name_list (List[Union[str, List[str]]]): A list of component names or groups of component names.
            params_config (Dict[str, ParamItem]): A dictionary of parameter configurations.
        """
        self.component_group_list = []
        self.gradio_input_dict = {}
        for component_name in component_name_list:
            if isinstance(component_name, list):
                self.component_group_list.append([params_config[name] for name in component_name])
            else:
                self.component_group_list.append([params_config[component_name]])

    def add_input(self, name: str, input: Any):
        """
        Adds an input to the gradio_input_dict.

        Args:
            name (str): The name of the input.
            input (Any): The input object.
        """
        self.gradio_input_dict[name] = input

    def build_group(self, component_list: List[ParamItem]):
        """
        Builds a group of components.

        Args:
            component_list (List[ParamItem]): A list of component items.
        """
        for component in component_list:
            if component.component_type == "audio":
                new_audio = gr.Audio(label=i18n(component.label),type="filepath")
                self.add_input(component.name, new_audio)
            elif component.type == "str":
                if component.choices is not None and len(component.choices) > 0:
                    choices = [[i18n(option), option] for option in component.choices]
                    new_dropdown = gr.Dropdown(label=i18n(component.label), choices=choices, value=component.default)
                    self.add_input(component.name, new_dropdown)
                else:
                    new_textbox = gr.Textbox(label=i18n(component.label), value=component.default)
                    self.add_input(component.name, new_textbox)
            elif component.type in ["int", "float"]:
                if component.min_value is not None and component.max_value is not None:
                    new_slider = gr.Slider(label=i18n(component.label), minimum=component.min_value, maximum=component.max_value, step=component.step, value=component.default)
                    self.add_input(component.name, new_slider)
                else:
                    new_number = gr.Number(label=i18n(component.label), value=component.default)
                    self.add_input(component.name, new_number)
            elif component.type == "bool":
                new_checkbox = gr.Checkbox(label=i18n(component.label), value=component.default)
                self.add_input(component.name, new_checkbox)
    

    
    def build(self):
            """
            Builds the Gradio input dictionary by iterating over the component group list
            and calling the build_group method for each group.

            Returns:
                dict: The Gradio input dictionary.
            """
            for group in self.component_group_list:
                with gr.Group():
                    self.build_group(group)
            return self.gradio_input_dict
            

def emit_on_change(*data):
    """
    Emits an on_change event for the Gradio inputs.

    Args:
        data (Any): The data to emit.
    """
    pass    

def register_on_change(component_name_list: List[Union[str, List[str]]], all_gradio_components: Dict[str, Any]):
    """
    Registers an on_change function for the Gradio inputs.

    Args:
        component_name_list (List[Union[str, List[str]]]): A list of component names or groups of component names.
        gradio_input_dict (Dict[str, Any]): The Gradio input dictionary.
        on_change (Optional[Callable[[Dict[str, Any]], None]]): The on_change function to register.
    """
    new_name_list = []
    
    for component_name in component_name_list:
        if isinstance(component_name, list):
            new_name_list = new_name_list.extend(component_name)
        else:
            new_name_list.append(component_name)
    all_components_list = [value for key, value in all_gradio_components.items()]

    param_dict = {
        "fn": emit_on_change,
        "inputs": all_components_list,
        "outputs": all_components_list
    }
    for name in new_name_list:
        gradio_component = all_gradio_components[name]
        if isinstance(gradio_component, gr.Textbox):
            gradio_component.blur(**param_dict)
        else:
            gradio_component.change(**param_dict)
            
    