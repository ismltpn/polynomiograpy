import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional
import polynomiograpy
from polynomiograpy.iterations import available_methods
from PIL import Image
import subprocess


class Tags:
    primary_window = "Primary"
    polynomial_preview = "PolynomialPreview"
    polynomial_raw_str = "PolynomialRawStr"
    is_polynomial_valid = "IsPolynomialValid"
    preview_image = "PreviewImage"
    error_field = "ErrorField"
    generate_output_modal = "GenerateOutputModal"
    generate_output_modal_text = "GenerateOutputModalText"
    generate_output_modal_button = "GenerateOutputModalButton"
    is_r_channel_active = "IsRChannelActive"
    is_g_channel_active = "IsGChannelActive"
    is_b_channel_active = "IsBChannelActive"
    # Values
    max_imag_value = "MaxImagValue"
    min_imag_value = "MinImagValue"
    max_real_value = "MaxRealValue"
    min_real_value = "MinRealValue"
    method_value = "MethodValue"
    color_value = "ColorValue"
    delta_r_value = "DeltaRValue"
    max_iter_r_value = "MaxIterRValue"
    delta_g_value = "DeltaGValue"
    max_iter_g_value = "MaxIterGValue"
    delta_b_value = "DeltaBValue"
    max_iter_b_value = "MaxIterBValue"
    width_value = "WidthValue"
    height_value = "HeightValue"
    filename_value = "FilenameValue"


if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(
        title="PolynomiograPy",
        width=800,
        height=500,
        resizable=False,
    )
    with dpg.font_registry():
        with dpg.font("gui_assets/fonts/LexendExa-Regular.ttf", 13) as default_font:
            # add the default font range
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range(0x2070, 0x209F)

    dpg.bind_font(default_font)
    im1 = Image.open(r"pp2preview_test.png").convert("RGBA")
    data = np.asfarray(im1, dtype="f")
    # raw_data = np.true_divide(data, 255.0)
    raw_data = np.ones((100, 100, 4), dtype=np.float32)

    # raw_data.resize((200, 200, 4))
    preview_screen = np.zeros((100, 100, 3), np.uint8)
    preview_screen_buffer = np.zeros((100, 100, 3), np.complex128)

    def update_dynamic_texture(sender, app_data: str, user_data: Optional[list[int]]):
        try:
            coefs_raw_str = dpg.get_value(Tags.polynomial_raw_str)
            coefs = (
                user_data if user_data else [float(x) for x in coefs_raw_str.split()]
            )
            max_imag = float(dpg.get_value(Tags.max_imag_value))
            min_imag = float(dpg.get_value(Tags.min_imag_value))
            max_real = float(dpg.get_value(Tags.max_real_value))
            min_real = float(dpg.get_value(Tags.min_real_value))
            width = 100
            height = 100
            scale_x = (max_real - min_real) / width
            scale_y = (max_imag - min_imag) / height
            shift_x = (max_real + min_real) / 2
            shift_y = (max_imag + min_imag) / 2
            # channels = {
            #     "red": 0,
            #     "green": 1,
            #     "blue": 2,
            # }
            # channel = channels[dpg.get_value(Tags.color_value)]
            preview_screen_buffer.fill(0)
            active = [
                dpg.get_value(Tags.is_r_channel_active),
                dpg.get_value(Tags.is_g_channel_active),
                dpg.get_value(Tags.is_b_channel_active),
            ]
            deltas = [
                float(dpg.get_value(Tags.delta_r_value)),
                float(dpg.get_value(Tags.delta_g_value)),
                float(dpg.get_value(Tags.delta_b_value)),
            ]
            max_iters = [
                int(dpg.get_value(Tags.max_iter_r_value)),
                int(dpg.get_value(Tags.max_iter_g_value)),
                int(dpg.get_value(Tags.max_iter_b_value)),
            ]
            for channel in [0, 1, 2]:
                if active[channel]:
                    polynomiograpy.compute_screen_for_single_poly(
                        dpg.get_value(Tags.method_value),
                        polynomiograpy.Polynomial(coeffs=coefs),
                        delta=deltas[channel],
                        width=width,
                        height=height,
                        screen=preview_screen,
                        screen_buffer=preview_screen_buffer,
                        max_value=max_iters[channel],
                        scale_x=scale_x,
                        scale_y=scale_y,
                        shift_x=shift_x,
                        shift_y=shift_y,
                        channel=channel,
                    )
            raw_data[:, :, :3] = np.true_divide(preview_screen, 255.0)
            dpg.set_value(Tags.error_field, "")
        except ValueError as e:
            raw_data.fill(1)
            dpg.set_value(Tags.error_field, str(e))
            import traceback

            traceback.print_exc()
        except Exception as e:
            raw_data.fill(1)
            dpg.set_value(Tags.error_field, str(e))
            import traceback

            traceback.print_exc()

    def generate_output(
        sender,
        app_data: str,
        user_data,
    ):
        try:
            dpg.set_value(Tags.generate_output_modal_text, "Generating...")
            dpg.configure_item(Tags.generate_output_modal_button, show=False)
            dpg.configure_item(Tags.generate_output_modal, show=True)
            coefs_raw_str = dpg.get_value(Tags.polynomial_raw_str)
            coefs = [float(x) for x in coefs_raw_str.split()]
            max_imag = float(dpg.get_value(Tags.max_imag_value))
            min_imag = float(dpg.get_value(Tags.min_imag_value))
            max_real = float(dpg.get_value(Tags.max_real_value))
            min_real = float(dpg.get_value(Tags.min_real_value))
            # delta = float(dpg.get_value(Tags.delta_value))
            # max_iter = int(dpg.get_value(Tags.max_iter_value))
            width = int(dpg.get_value(Tags.width_value))
            height = int(dpg.get_value(Tags.height_value))
            scale_x = (max_real - min_real) / width
            scale_y = (max_imag - min_imag) / height
            shift_x = (max_real + min_real) / 2
            shift_y = (max_imag + min_imag) / 2
            channels = {
                "red": 0,
                "green": 1,
                "blue": 2,
            }
            channel = channels[dpg.get_value(Tags.color_value)]
            output_screen = np.zeros((height, width, 3), np.uint8)
            output_screen_buffer = np.zeros((height, width, 3), np.complex128)
            filename = dpg.get_value(Tags.filename_value)
            active = [
                dpg.get_value(Tags.is_r_channel_active),
                dpg.get_value(Tags.is_g_channel_active),
                dpg.get_value(Tags.is_b_channel_active),
            ]
            deltas = [
                float(dpg.get_value(Tags.delta_r_value)),
                float(dpg.get_value(Tags.delta_g_value)),
                float(dpg.get_value(Tags.delta_b_value)),
            ]
            max_iters = [
                int(dpg.get_value(Tags.max_iter_r_value)),
                int(dpg.get_value(Tags.max_iter_g_value)),
                int(dpg.get_value(Tags.max_iter_b_value)),
            ]
            for channel in [0, 1, 2]:
                if active[channel]:
                    polynomiograpy.compute_screen_for_single_poly(
                        dpg.get_value(Tags.method_value),
                        polynomiograpy.Polynomial(coeffs=coefs),
                        delta=deltas[channel],
                        width=width,
                        height=height,
                        screen=output_screen,
                        screen_buffer=output_screen_buffer,
                        max_value=max_iters[channel],
                        scale_x=scale_x,
                        scale_y=scale_y,
                        shift_x=shift_x,
                        shift_y=shift_y,
                        channel=channel,
                    )
            im = Image.fromarray(output_screen, mode="RGB")
            im.save(filename, format="PNG")
            dpg.set_value(Tags.generate_output_modal_text, f"Done. Saved to {filename}")
            dpg.configure_item(Tags.generate_output_modal_button, show=True)
            subprocess.call(("open", filename))
        except Exception as e:
            dpg.set_value(Tags.generate_output_modal_text, f"Error: {e}")
            dpg.configure_item(Tags.generate_output_modal_button, show=True)

    def polynomial_text_field_callback(
        sender,
        app_data: str,
        user_data,
    ):
        try:
            coefs = [float(x) for x in app_data.split()]
            dpg.set_value(
                Tags.polynomial_preview, str(np.polynomial.Polynomial(coef=coefs))
            )
            # dpg.set_clipboard_text(text=str(np.polynomial.Polynomial(coef=coefs)))
            # dpg.set_value(Tags.polynomial_raw_str, app_data)
            update_dynamic_texture(sender, app_data, coefs)
        except ValueError:
            dpg.set_value(Tags.polynomial_preview, "Invalid Polynomial")
            dpg.set_value(Tags.is_polynomial_valid, False)

    with dpg.value_registry():
        dpg.add_bool_value(default_value=True, tag=Tags.is_polynomial_valid)
        dpg.add_string_value(default_value="1 0 1 1", tag=Tags.polynomial_raw_str)
        dpg.add_string_value(default_value="3", tag=Tags.max_imag_value)
        dpg.add_string_value(default_value="-3", tag=Tags.min_imag_value)
        dpg.add_string_value(default_value="3", tag=Tags.max_real_value)
        dpg.add_string_value(default_value="-3", tag=Tags.min_real_value)
        dpg.add_string_value(
            default_value="inverse_interpolation", tag=Tags.method_value
        )
        dpg.add_string_value(default_value="red", tag=Tags.color_value)
        dpg.add_bool_value(default_value=True, tag=Tags.is_r_channel_active)
        dpg.add_string_value(default_value="0.1", tag=Tags.delta_r_value)
        dpg.add_string_value(default_value="16", tag=Tags.max_iter_r_value)
        dpg.add_bool_value(default_value=False, tag=Tags.is_g_channel_active)
        dpg.add_string_value(default_value="0.05", tag=Tags.delta_g_value)
        dpg.add_string_value(default_value="16", tag=Tags.max_iter_g_value)
        dpg.add_bool_value(default_value=True, tag=Tags.is_b_channel_active)
        dpg.add_string_value(default_value="0.1", tag=Tags.delta_b_value)
        dpg.add_string_value(default_value="8", tag=Tags.max_iter_b_value)
        dpg.add_string_value(default_value="1000", tag=Tags.width_value)
        dpg.add_string_value(default_value="1000", tag=Tags.height_value)
        dpg.add_string_value(default_value="out.png", tag=Tags.filename_value)

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(
            width=100,
            height=100,
            default_value=raw_data,
            format=dpg.mvFormat_Float_rgba,
            tag=Tags.preview_image,
        )

    with dpg.window(
        label="Generate Output",
        modal=True,
        show=False,
        tag=Tags.generate_output_modal,
        no_title_bar=True,
        height=30,
        autosize=True,
    ):
        dpg.add_text("Generating", tag=Tags.generate_output_modal_text)
        dpg.add_button(
            label="OK",
            width=75,
            callback=lambda: dpg.configure_item(Tags.generate_output_modal, show=False),
            show=False,
            tag=Tags.generate_output_modal_button,
        )

    with dpg.window(tag=Tags.primary_window):
        with dpg.menu_bar():
            with dpg.menu(label="Mode"):
                dpg.add_menu_item(label="Iterative")
                dpg.add_menu_item(label="Finite Field")

        with dpg.table(
            header_row=False,
            # resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
            borders_outerH=True,
            borders_innerV=True,
            borders_innerH=True,
            borders_outerV=True,
        ):
            dpg.add_table_column()
            with dpg.table_row():
                with dpg.table(
                    header_row=False,
                    # resizable=True,
                    policy=dpg.mvTable_SizingStretchProp,
                    borders_innerH=True,
                ):
                    dpg.add_table_column()
                    dpg.add_table_column(
                        width_fixed=True,
                        init_width_or_weight=400,
                    )

                    with dpg.table_row():
                        with dpg.table(
                            header_row=False,
                            # resizable=True,
                            borders_innerV=True,
                            borders_outerV=True,
                            borders_outerH=True,
                        ):
                            dpg.add_table_column()
                            with dpg.table_row():
                                with dpg.table_cell():
                                    dpg.add_text("Input Polynomial:")
                                    dpg.add_input_text(
                                        source=Tags.polynomial_raw_str,
                                        callback=polynomial_text_field_callback,
                                    )
                                    dpg.add_text(
                                        "1.0 + 0.0·x + 1.0·x² + 1.0·x³",
                                        tag=Tags.polynomial_preview,
                                    )
                            with dpg.table_row():
                                with dpg.table_cell():
                                    dpg.add_text("Method:")
                                    dpg.add_combo(
                                        items=list(available_methods),
                                        source=Tags.method_value,
                                        callback=update_dynamic_texture,
                                    )
                                    with dpg.table(
                                        header_row=False,
                                        # resizable=True,
                                    ):
                                        dpg.add_table_column(width_stretch=True)
                                        dpg.add_table_column(width_stretch=True)
                                        dpg.add_table_column()
                                        with dpg.table_row():
                                            dpg.add_checkbox(
                                                label="R",
                                                source=Tags.is_r_channel_active,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Delta",
                                                source=Tags.delta_r_value,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Max Iter",
                                                source=Tags.max_iter_r_value,
                                                callback=update_dynamic_texture,
                                            )
                                        with dpg.table_row():
                                            dpg.add_checkbox(
                                                label="G",
                                                source=Tags.is_g_channel_active,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Delta",
                                                source=Tags.delta_g_value,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Max Iter",
                                                source=Tags.max_iter_g_value,
                                                callback=update_dynamic_texture,
                                            )
                                        with dpg.table_row():
                                            dpg.add_checkbox(
                                                label="B",
                                                source=Tags.is_b_channel_active,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Delta",
                                                source=Tags.delta_b_value,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Max Iter",
                                                source=Tags.max_iter_b_value,
                                                callback=update_dynamic_texture,
                                            )
                        with dpg.table(
                            header_row=False,
                            borders_outerV=True,
                            borders_outerH=True,
                        ):
                            dpg.add_table_column()
                            with dpg.table_row():
                                with dpg.table_cell():
                                    dpg.add_text("Preview")
                                    dpg.add_image(
                                        Tags.preview_image, width=300, height=300
                                    )
                                    dpg.add_text(
                                        "Initial",
                                        tag=Tags.error_field,
                                    )
                                    with dpg.table(header_row=False):
                                        dpg.add_table_column()
                                        dpg.add_table_column()
                                        with dpg.table_row():
                                            dpg.add_input_text(
                                                label="Max Imag",
                                                source=Tags.max_imag_value,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Max Real",
                                                source=Tags.max_real_value,
                                                callback=update_dynamic_texture,
                                            )
                                        with dpg.table_row():
                                            dpg.add_input_text(
                                                label="Min Imag",
                                                source=Tags.min_imag_value,
                                                callback=update_dynamic_texture,
                                            )
                                            dpg.add_input_text(
                                                label="Min Real",
                                                source=Tags.min_real_value,
                                                callback=update_dynamic_texture,
                                            )

            with dpg.table_row(height=50):
                with dpg.table(
                    header_row=False,
                ):
                    dpg.add_table_column(width=400)
                    dpg.add_table_column(width=400)
                    with dpg.table_row():
                        with dpg.table_cell():
                            with dpg.table(
                                header_row=False,
                                no_pad_outerX=True,
                                no_pad_innerX=True,
                                pad_outerX=False,
                            ):
                                dpg.add_table_column(width_stretch=True)
                                dpg.add_table_column(width_stretch=True)
                                with dpg.table_row():
                                    dpg.add_input_text(
                                        label="Width",
                                        width=100,
                                        source=Tags.width_value,
                                    )
                                    dpg.add_input_text(
                                        label="Height",
                                        width=100,
                                        source=Tags.height_value,
                                    )
                            dpg.add_input_text(
                                label="File name", width=300, source=Tags.filename_value
                            )
                        dpg.add_button(
                            label="Generate",
                            width=390,
                            height=48,
                            callback=generate_output,
                        )

    # initial load
    update_dynamic_texture(None, "", None)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(Tags.primary_window, True)
    dpg.start_dearpygui()
    dpg.destroy_context()
