#----------------------------------------------------------------------------
# Create Date: 13/07/2024 5:30AM
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Davi"

class Model1D:

    def __init__(self, Nz: int, vp_interfaces: list, rho_interfaces: list, interfaces: list) -> None:
        ErrorHandling.negative_model_parameters(Nz)

        self.Nz = Nz
        self.vp_interfaces = vp_interfaces
        self.rho_interfaces = rho_interfaces
        self.interfaces = interfaces

    def set_model_to_plot(self, value_interface) -> np.array:
        self.create_zero_model()
        self.__initial_condition(value_interface)

        ErrorHandling.interface_error(self.interfaces, value_interface)

        return self.__create_model_loop(value_interface)

    def create_zero_model(self) -> None:
        self.model_to_plot = np.zeros(self.Nz)

    def __initial_condition(self, value_interface) -> None:
        self.model_to_plot[:interfaces[0]] = value_interface[0]

    def __create_model_loop(self, current_model_interface: list) -> np.array:
        for layer, property_value in enumerate(current_model_interface[1:]):
            self.model_to_plot[self.interfaces[layer]:] = property_value
        return self.model_to_plot        


class RickerWavelet:

    def __init__(self, Nt: int, dt: float, fmax: int) -> None:
        self.Nt = Nt
        self.dt = dt
        self.fmax = fmax
        
    def set_wavelet(self) -> np.array:
        self.t = self.__set_wavelet_time()
        arg = self.__set_wavelet_arg()
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    def __set_wavelet_time(self):
        t0 = 2 * np.pi / self.fmax
        return np.arange(self.Nt) * self.dt - t0

    def __set_wavelet_arg(self):
        return (np.pi * self.fmax * self.t) ** 2


class OrmsbyWavelet:

    def __init__(self, Nt: int, dt: int, ormsby_frequencies: list, time_lag: int):
        self.ormsby_frequencies = ormsby_frequencies
        self.Nt = Nt
        self.dt = dt
        self.time_lag = time_lag
    
    def set_wavelet(self) -> np.array:
        self.t = self.__set_wavelet_time()
        args = self.__set_wavelet_arg(self.ormsby_frequencies)
        return (args[0] - args[1]) - (args[2] - args[3])

    def __set_wavelet_time(self):
        return np.arange(self.Nt) * self.dt - self.time_lag

    def __set_wavelet_arg(self, freqs: int) -> np.array:
        arg1 = np.sinc(np.pi*freqs[3]*self.t)**2 * (np.pi*freqs[3])**2 / (np.pi*freqs[3] - np.pi*freqs[2])
        arg2 = np.sinc(np.pi*freqs[2]*self.t)**2 * (np.pi*freqs[2])**2 / (np.pi*freqs[3] - np.pi*freqs[2])
        arg3 = np.sinc(np.pi*freqs[1]*self.t)**2 * (np.pi*freqs[1])**2 / (np.pi*freqs[1] - np.pi*freqs[0])
        arg4 = np.sinc(np.pi*freqs[0]*self.t)**2 * (np.pi*freqs[0])**2 / (np.pi*freqs[1] - np.pi*freqs[0])
        return [arg1, arg2, arg3, arg4]


class Convolution:
    """
    Convolves a input signal(a wavelet) with a filter(reflectivity function)
    """

    def __init__(self, filter_kernel: list, input_signal: list) -> None:
        self.filter_kernel = filter_kernel
        self.input_signal = input_signal
    
    def get_convolution(self) -> np.array:
        return np.convolve(self.input_signal, self.filter_kernel, mode='same') 


class PlotModel:
    """
    Plots 1D Model Properties
    """

    def __init__(self, model_vp: list, model_rho: list, depth: list):
        self.model_vp = model_vp
        self.model_rho = model_rho
        self.depth = depth

    def model_plot(self, ax, id):
        ax[id].plot(self.model_vp, self.depth, label="Velocity [m/s]")
        ax[id].plot(self.model_rho, self.depth, color="orange", label="Density [kg/$m^3$]")

        self.__model_title(ax, id)
        self.__model_legend(ax, id)

    def __model_title(self, ax, id: int) -> None:
        ax[id].set_title("Model Properties", fontsize=13)

    def __model_legend(self, ax, id: int) -> None:
        ax[id].legend(loc="lower right")


class PlotConvolution:
    """
    Plots for better visualizing convolution model
    """

    def __init__(self, model_obj, lithology_obj, wavelet_obj, acoustic_impendance: list, reflectivity_function: list, convolved_signal: list) -> None:
        self.acoustic_impendance = acoustic_impendance
        self.reflectivity_function = reflectivity_function
        self.wavelet_obj = wavelet_obj
        self.convolved_signal = convolved_signal
        self.lithology_obj = lithology_obj
        self.model_obj = model_obj

    def model_plot(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=6, figsize=(18,10))

        self.lithology_obj.model_plot(self.ax, id=0, func_size=len(self.reflectivity_function))
        self.model_obj.model_plot(self.ax, id=1)

        self.ax[2].step(self.acoustic_impendance, np.arange(len(self.acoustic_impendance)))
        self.ax[2].set_title("Acoustic Impedance", fontsize=13)

        self.ax[3].plot(self.reflectivity_function, np.arange(len(self.reflectivity_function)))
        self.ax[3].set_title("Reflectivity Function", fontsize=13)

        self.ax[4].plot(self.__get_wavelet(), np.arange(len(self.__get_wavelet())))
        self.ax[4].set_title("Wavelet", fontsize=13)

        self.ax[5].plot(self.convolved_signal, np.arange(len(self.convolved_signal)))
        self.ax[5].set_title("Convolved Signal", fontsize=13)

        self.__get_plot_grid()

        plt.show()
        return self.fig
  
    def __get_plot_grid(self) -> None:
        for i in range(len(self.ax)):
            self.ax[i].grid(True)       

    def __get_wavelet(self) -> None:
       return self.wavelet_obj.set_wavelet()


class PlotLithology:
    """
    Plots a lithological profile
    """

    def __init__(self, lithology_types: list[str], lithology_respective_color: dict, interfaces: list) -> None:
        self.lithology_types = lithology_types
        self.lithology_respective_color = lithology_respective_color
        self.interfaces = interfaces

    def model_plot(self, ax, id: int, func_size: int) -> None:
        self.ax = ax
        for i, layer in enumerate(reversed(self.lithology_types)):
            if i == 0:
                ax[id].fill_betweenx(np.arange(0, self.interfaces[i]), 0, 1, color=self.lithology_respective_color[layer], label=layer)
            elif i < len(self.interfaces):
                ax[id].fill_betweenx(np.arange(self.interfaces[i-1], self.interfaces[i]), 0, 1, color=self.lithology_respective_color[layer], label=layer)
            else:
                ax[id].fill_betweenx(np.arange(self.interfaces[-1], func_size), 0, 1, color=self.lithology_respective_color[layer], label=layer)

        self.__model_ticks(id, func_size)
        self.__model_legends(id)
        self.__model_labels(id, func_size)
        ax[id].set_title("Lithological Profile", fontsize=13)

    def __model_ticks(self, id: int, func_size: int) -> None:
        self.ax[id].set_xticks([])
        self.ax[id].set_yticks(np.linspace(0, func_size, 6))

    def __model_labels(self, id: int, func_size: int) -> None:
        self.ax[id].set_yticklabels(np.linspace(func_size-1, 0, 6).astype(int))

    def __model_legends(self, id: int) -> None:
        self.ax[id].legend(loc='lower left')


class AuxFunctions:
    """
    Class for auxiliar functions
    """

    @staticmethod
    def reflection_coefficient(Nz: int, interfaces: list, acoustic_impedance: list) -> np.array:
        reflectivity_function = np.zeros(Nz)
        for i in range(len(acoustic_impedance) - 1):
            reflectivity_function[interfaces[i]] = (acoustic_impedance[i+1] - acoustic_impedance[i]) / (acoustic_impedance[i+1] + acoustic_impedance[i])
        return reflectivity_function

    @staticmethod
    def get_convolution_to_plot(acoustic_impedance: list, interfaces: list, Nz: int) -> np.array:
        acoustic_impedance_to_plot = np.zeros(Nz)
        for layer, property_value in enumerate(acoustic_impedance[1:]):
            acoustic_impedance_to_plot[interfaces[layer]:] = property_value
        return acoustic_impedance_to_plot


class ErrorHandling:
    """
    Class for management of error messages
    """

    @staticmethod
    def interface_error(interfaces: list, value_interface: list) -> None:
        if len(interfaces) != len(value_interface) - 1:
            raise ValueError("Interfaces Must be a Length Smaller than velocity_interfaces!")

    @staticmethod
    def negative_model_parameters(Nz: int) -> None:
        if Nz < 0:
            raise ValueError("Model Parameters Cannot be Negative!")

    @staticmethod
    def wrong_key(available_keys: dict):
        print(f"Please use the availables keys: {available_keys}")
        exit()


# Model Parameters
Nz = 501
Nt = 501

dt = 0.001  # 1ms
dz = 10     # m
fmax = 25  # Hz
ormsby_frequencies = [5, 10, 40, 45] #[hz]

interfaces = [100, 300, 400]
vp_interfaces = [3000, 4500, 2500, 3600]
rho_interfaces = [1900, 2100, 2700, 2600]
depth = np.arange(Nz) * dz

# Create Wavelet to Plot
instanced_ricker_wavelet = RickerWavelet(Nt, dt, fmax)
instaced_ormsby_wavelet = OrmsbyWavelet(Nt, dt, ormsby_frequencies, time_lag=0.3)
ricker_wavelet = instanced_ricker_wavelet.set_wavelet()
ormsby_wavelet = instaced_ormsby_wavelet.set_wavelet()

# Convolutional Model
acoustic_impedance = [rho_interfaces[i] * vp_interfaces[i] for i in range(len(vp_interfaces))]
acoustic_impedance_to_plot = AuxFunctions.get_convolution_to_plot(acoustic_impedance, interfaces, Nz)
reflectivity_function = AuxFunctions.reflection_coefficient(Nz, interfaces, acoustic_impedance)
convolved_signal = Convolution(reflectivity_function, ricker_wavelet).get_convolution()

# Create Model to Plot
instanced_model = Model1D(Nz, vp_interfaces, rho_interfaces, interfaces)
model_vp = instanced_model.set_model_to_plot(vp_interfaces)
model_rho = instanced_model.set_model_to_plot(rho_interfaces)

# Plot Model 1D
instanced_model_plot = PlotModel(model_vp, model_rho, depth)
#model_plot = instanced_model_plot.model_plot()

# Lithology Parameters
lithology_type = {'sandstone': 'yellow', 'dolomite': 'slateblue', 'carbonate': 'blue', 'salt': 'lightgray'}
lithology_order = ['sandstone', 'salt', 'carbonate', 'dolomite']

# Create Lithology Model
instanced_lithology_plot = PlotLithology(lithology_order, lithology_type, interfaces)

# Plot Convolutional Model
instanced_convolutional_plot = PlotConvolution(instanced_model_plot, instanced_lithology_plot, instanced_ricker_wavelet, \
                                                acoustic_impedance_to_plot, reflectivity_function, convolved_signal)
convolution_plot = instanced_convolutional_plot.model_plot()
