################################################################################
#
# fractalgui2.py - Toren Wallengren
#
# This program constructs a gui to explore escape-time fractals (such as the
# mandelbrot set).
#
################################################################################
#
# Libraries to import

# Import tkinter and related library information
import tkinter as tk
from tkinter import ttk

# Functions to import from PIL (to help convert an RGB image into a form Tkinter
# can use
from PIL import Image, ImageTk

# Import numpy (library for array methods similar to MATLAB)
import numpy as np

# Import matplotlib and related library information
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################################################################################
#
# Initialize global variables

    ############################################################################
    #
    # Styles

LARGE_FONT = ("Verdana",12)
style.use("classic")

    ############################################################################
    #
    # Embedded Color Dictionaries and specified default

# Everyone Is Better Than Everyone Color Palette
everyoneisbetterthaneveryone = {}
everyoneisbetterthaneveryone[0] = [206, 181, 182, 169, 180, 169, 182, 181, 206]
everyoneisbetterthaneveryone[1] = [195, 139, 128, 99, 64, 99, 128, 139, 195]
everyoneisbetterthaneveryone[2] = [231, 210, 189, 156, 146, 156, 189, 210, 231]
everyoneisbetterthaneveryone[3] = [-127.5, -95.625, -63.75, -31.875, 0, 31.875, 63.75, 95.625, 127.5]

# That Hurt My Friend Color Palette
thathurtmyfriend = {}
thathurtmyfriend[0] = [189, 159, 144, 126, 114, 126, 144, 159, 189]
thathurtmyfriend[1] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
thathurtmyfriend[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
thathurtmyfriend[3] = [-127.5, -95.625, -63.75, -31.875, 0, 31.875, 63.75, 95.625, 127.5]

# Unicorn Skin Color Palette
unicornskin = {}
unicornskin[0] = [242, 109, 246, 138, 217, 138, 246, 109, 242]
unicornskin[1] = [131, 197, 246, 255, 142, 255, 246, 197, 131]
unicornskin[2] = [209, 251, 140, 164, 251, 164, 140, 251, 209]
unicornskin[3] = [-127.5, -95.625, -63.75, -31.875, 0, 31.875, 63.75, 95.625, 127.5]

# Default Color Dictionary
defaultcolordict = unicornskin

    ############################################################################
    #
    # Default Values

# Default Pixel x Pixel size of fractal
defaultsize = 300

# Default x-range for color selection figure
xrange = np.linspace(-127.5,127.5,1000)

# Fourier Precision
fourierprecision = 10

# Escape condition/bound
bound = 2

# Color Multiplier
colormultiplier = 3

# Colorshift
colorshift = 0

# Default zoom scale
zoom = 5

    ############################################################################
    #
    # zdict Initialization

xo = 0 # Center of x-coordinates
yo = 0 # Center of y-coordinates
x_offset = 2 # Max horizontal distance from xo
y_offset = x_offset # Max vertical distance from yo
x = np.linspace(xo-x_offset, xo+x_offset, defaultsize) # Range of x-coordinates
y = np.linspace(yo-y_offset, yo+y_offset, defaultsize) # Range of y-coordinates
zo = np.zeros([defaultsize,defaultsize]) # Initial z-value for iterated function
X, Y = np.meshgrid(x,y) # Create grid of (x,y) coordinates
Z = X + 1j*Y # Create points of the complex plane with (x,y) coordinates
zdict = {} # Initialize dictionary for points on the complex plane
zdict[0] = Z # Store the complex plane
zdict[1] = np.absolute(Z) # Store corresponding magnitudes
zdict[2] = np.zeros([defaultsize,defaultsize,3], dtype=np.uint8) # Store RGB values (init Black)
zdict[3] = np.zeros([defaultsize,defaultsize]) # Number of iterations to escape
zdict[4] = np.zeros([defaultsize,defaultsize]) # 1 if escape, 0 if not
count = 0

################################################################################
#
# Fourier Coefficients Function

def fouriercoef(colorscheme,precision):

    Rrange = colorscheme[0]
    Grange = colorscheme[1]
    Brange = colorscheme[2]
    Xrange = colorscheme[3]

    N = np.size(Xrange)

    slopes = np.zeros([3,N-1])

    ao = np.zeros([3,1])
    an = np.zeros([3,precision])
    bn = np.zeros([3,precision])

    for i in range(0,N-1):

        slopes[0,i] = (Rrange[i+1] - Rrange[i])/(Xrange[i+1] - Xrange[i])
        slopes[1,i] = (Grange[i+1] - Grange[i])/(Xrange[i+1] - Xrange[i])
        slopes[2,i] = (Brange[i+1] - Brange[i])/(Xrange[i+1] - Xrange[i])

        ao[0] = ao[0] + (1/2)*slopes[0,i]*(Xrange[i+1]**2-Xrange[i]**2) + (Rrange[i] - slopes[0,i]*Xrange[i])*(Xrange[i+1] - Xrange[i])
        ao[1] = ao[1] + (1/2)*slopes[1,i]*(Xrange[i+1]**2-Xrange[i]**2) + (Grange[i] - slopes[1,i]*Xrange[i])*(Xrange[i+1] - Xrange[i])
        ao[2] = ao[2] + (1/2)*slopes[2,i]*(Xrange[i+1]**2-Xrange[i]**2) + (Brange[i] - slopes[2,i]*Xrange[i])*(Xrange[i+1] - Xrange[i])

    ao[0] = 2*ao[0]/255
    ao[1] = 2*ao[1]/255
    ao[2] = 2*ao[2]/255

    for n in range(1,precision):

        for i in range(0,N-1):

            an[0,n] = an[0,n] + 127.5*slopes[0,i]*(np.cos(n*np.pi*Xrange[i+1]/127.5) - np.cos(n*np.pi*Xrange[i]/127.5))/(n*np.pi) + (slopes[0,i]*(Xrange[i+1] - Xrange[i]) + Rrange[i])*np.sin(n*np.pi*Xrange[i+1]/127.5) - Rrange[i]*np.sin(n*np.pi*Xrange[i]/127.5)
            an[1,n] = an[1,n] + 127.5*slopes[1,i]*(np.cos(n*np.pi*Xrange[i+1]/127.5) - np.cos(n*np.pi*Xrange[i]/127.5))/(n*np.pi) + (slopes[1,i]*(Xrange[i+1] - Xrange[i]) + Grange[i])*np.sin(n*np.pi*Xrange[i+1]/127.5) - Grange[i]*np.sin(n*np.pi*Xrange[i]/127.5)
            an[2,n] = an[2,n] + 127.5*slopes[2,i]*(np.cos(n*np.pi*Xrange[i+1]/127.5) - np.cos(n*np.pi*Xrange[i]/127.5))/(n*np.pi) + (slopes[2,i]*(Xrange[i+1] - Xrange[i]) + Brange[i])*np.sin(n*np.pi*Xrange[i+1]/127.5) - Brange[i]*np.sin(n*np.pi*Xrange[i]/127.5)

            bn[0,n] = bn[0,n] + 127.5*slopes[0,i]*(np.sin(n*np.pi*Xrange[i+1]/127.5) - np.sin(n*np.pi*Xrange[i]/127.5))/(n*np.pi) - (slopes[0,i]*(Xrange[i+1] - Xrange[i]) + Rrange[i])*np.cos(n*np.pi*Xrange[i+1]/127.5) + Rrange[i]*np.cos(n*np.pi*Xrange[i]/127.5)
            bn[1,n] = bn[1,n] + 127.5*slopes[1,i]*(np.sin(n*np.pi*Xrange[i+1]/127.5) - np.sin(n*np.pi*Xrange[i]/127.5))/(n*np.pi) - (slopes[1,i]*(Xrange[i+1] - Xrange[i]) + Grange[i])*np.cos(n*np.pi*Xrange[i+1]/127.5) + Grange[i]*np.cos(n*np.pi*Xrange[i]/127.5)
            bn[2,n] = bn[2,n] + 127.5*slopes[2,i]*(np.sin(n*np.pi*Xrange[i+1]/127.5) - np.sin(n*np.pi*Xrange[i]/127.5))/(n*np.pi) - (slopes[2,i]*(Xrange[i+1] - Xrange[i]) + Brange[i])*np.cos(n*np.pi*Xrange[i+1]/127.5) + Brange[i]*np.cos(n*np.pi*Xrange[i]/127.5)

        an[0,n] = an[0,n]/(n*np.pi)
        an[1,n] = an[1,n]/(n*np.pi)
        an[2,n] = an[2,n]/(n*np.pi)

        bn[0,n] = bn[0,n]/(n*np.pi)
        bn[1,n] = bn[1,n]/(n*np.pi)
        bn[2,n] = bn[2,n]/(n*np.pi)

                                                                                                                                                                                                                                              
    return ao, an, bn

################################################################################
#
# Fourier Color Function

def colorfunction(Xrange,precision,ao,an,bn):
        
    Rvalues = 0
    Gvalues = 0
    Bvalues = 0

    for n in range(0,precision):

        Rvalues = Rvalues + an[0,n]*np.cos(2*np.pi*n*Xrange/255) + bn[0,n]*np.sin(2*np.pi*n*Xrange/255)
        Gvalues = Gvalues + an[1,n]*np.cos(2*np.pi*n*Xrange/255) + bn[1,n]*np.sin(2*np.pi*n*Xrange/255)
        Bvalues = Bvalues + an[2,n]*np.cos(2*np.pi*n*Xrange/255) + bn[2,n]*np.sin(2*np.pi*n*Xrange/255)

    Rvalues = Rvalues + ao[0]/2
    Gvalues = Gvalues + ao[1]/2
    Bvalues = Bvalues + ao[2]/2
        
    return Rvalues, Gvalues, Bvalues

################################################################################
#
# Update Picture function

def updatepicture(label):

    global count, z, Z, zo, zdict

    z = (zo)**(2) + Z

    zdict[1] = np.absolute(z) # Get magnitudes of values of current iteration

    info1 = zdict[1] # Store magnitude values in the variable info1
        
    info2 = zdict[2] # Store current RGB values in info2

    print(count+1)

    count = count + 1
    zo = z

    value0 = np.mod(colormultiplier*count + colorshift,256)

    [Rcurrent, Gcurrent, Bcurrent] = colorfunction(value0,fourierprecision,ao,an,bn)
    R = np.floor(Rcurrent)
    G = np.floor(Gcurrent)
    B = np.floor(Bcurrent)

    # NOTE: This next part of the program goes through the image pixel by
    # pixel to test for the escape condition. It is the most time-consuming
    # part of the program.

    # For each pixel column in the image
    for num1 in range(0,defaultsize):

        # For each row in the pixel column
        for num2 in range(0,defaultsize):
                
            test2 = info1[num1,num2]
            test3 = info2[num1,num2]

            if (test2 > bound) & (np.all(test3 == 0)):
                        
                    zdict[2][num1,num2] = [R,G,B]
                    zdict[3][num1,num2] = count
                    zdict[4][num1,num2] = 1

                    z[num1,num2] = 0
                    Z[num1,num2] = 0

    img = Image.fromarray(zdict[2], 'RGB')
    photo = ImageTk.PhotoImage(img)
    label.config(image=photo)
    label.image = photo


################################################################################
#
# Initialize figure for color selection page

# Create matplotlib figure
f = Figure()

# Add a subplot to the figure
a = f.add_subplot(111)

Rvalues = defaultcolordict[0]
Gvalues = defaultcolordict[1]
Bvalues = defaultcolordict[2]
Xvalues = defaultcolordict[3]

N = np.size(Xvalues)

ao, an, bn = fouriercoef(defaultcolordict,fourierprecision)
Rinit, Ginit, Binit = colorfunction(xrange,fourierprecision,ao,an,bn)

# Plot red values on the subplot       
a.plot(Xvalues, Rvalues, 'ro')
a.plot(Xvalues, Rvalues, 'r--')
a.plot(xrange, Rinit, 'r')

# Plot green values on the subplot
a.plot(Xvalues, Gvalues, 'go')
a.plot(Xvalues, Gvalues, 'g--')
a.plot(xrange, Ginit, 'g')

# Plot blut values on the subplot
a.plot(Xvalues, Bvalues, 'bo')
a.plot(Xvalues, Bvalues, 'b--')
a.plot(xrange, Binit, 'b')

a.axis([-127.5,127.5,0,255])

################################################################################
#
# Animate color selection approximation function

def animatecolorselection(i):

    currentvalue = np.mod(i,fourierprecision) + 1
    Rinit, Ginit, Binit = colorfunction(xrange,currentvalue,ao,an,bn)

    # Clear figure
    a.clear()

    # Plot red values on the subplot       
    a.plot(Xvalues,Rvalues,'ro')
    a.plot(Xvalues,Rvalues,'r--')
    a.plot(xrange, Rinit, 'r')

    # Plot green values on the subplot
    a.plot(Xvalues,Gvalues,'go')
    a.plot(Xvalues,Gvalues,'g--')
    a.plot(xrange, Ginit, 'g')

    # Plot blue values on the subplot
    a.plot(Xvalues,Bvalues,'bo')
    a.plot(Xvalues,Bvalues,'b--')
    a.plot(xrange, Binit, 'b')

    a.axis([-127.5,127.5,0,255])

################################################################################
#
# Class for the main app - Inherit functionality from tk.Tk


class FracWinApp(tk.Tk):

    ############################################################################
    #
    # Methods of FracWinApp class

    # Init method - Automatically called when an instance of FracWin is made
    def __init__(self, *args, **kwargs):

        ########################################################################
        #
        # Miscellaneous Window Initializations

        # Call init method from tk.Tk - Pass arguments of FracWin init method
        tk.Tk.__init__(self, *args, **kwargs)

        # Specify text at the top of the window for the entire app
        tk.Tk.wm_title(self, "Fractal Explorer")

        ########################################################################
        #
        # Window Container Specifications

        # Create a container to place objects in
        container = tk.Frame(self)

        # Pack container in window
        container.pack(side="top", fill="both", expand= True)

        # Specify grid configurations for container
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        ########################################################################
        #
        # Page Initialization

        # Initialize frame dictionary
        self.frames = {}

        # Populate frames dictionary with different pages for the app
        for F in (MainMenuPage, FractalWindow, SelectColorsPage):

            # Create current frame
            frame = F(container, self)

            # Store in frame dictionary with corresponding key
            self.frames[F] = frame

            # Specify location on the grid
            frame.grid(row=0, column=0, sticky="nsew")

        ########################################################################
        #
        # Display First Window

        # Show the Main Menu
        self.show_frame(MainMenuPage)

    ############################################################################

    # Define the show_frame method used in the __init__ method
    def show_frame(self, controller):

        # Frame specified by controller
        frame = self.frames[controller]

        # Raise frame
        frame.tkraise()

################################################################################
#
# Class for the Main Menu Page


class MainMenuPage(tk.Frame):

    ############################################################################
    #
    # Methods of MainMenuPage class

    # Init method - Automatically called when an instance of MainMenuPage is made
    def __init__(self, parent, controller):

        ########################################################################
        #
        # Frame Initialization

        # Call init method from tk.Frame
        tk.Frame.__init__(self,parent)

        ########################################################################
        #
        # Main Page Label

        # Create label
        label = tk.Label(self, text="Main Menu", font=LARGE_FONT, relief="solid",
                         width=100, height=3)

        # Pack the label in the frame
        label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Fractal Transition Button

        # Create start button to transition to fractal window
        startbutton = ttk.Button(self, text="Fractal Window",
                                command=lambda: controller.show_frame(FractalWindow))

        # Pack button into frame
        startbutton.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Color Transition Button

        # Create color button to transition to color select page
        colorbutton = ttk.Button(self, text="Select Colors",
                                command=lambda: controller.show_frame(SelectColorsPage))

        # Pack button into frame
        colorbutton.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

################################################################################
#
# Class for the Fractal Window Page


class FractalWindow(tk.Frame):

    ############################################################################
    #
    # Methods of FractalWindow class
    
    # Init method - Automatically called when an instance of Fractal is made
    def __init__(self, parent, controller):

        ########################################################################
        #
        # Frame Initialization

        # Call init method from tk.Frame
        tk.Frame.__init__(self,parent)

        ########################################################################
        #
        # Main Label

        img = Image.fromarray(zdict[2], 'RGB')
        photo = ImageTk.PhotoImage(img)

        # Create label
        self.label = tk.Label(self, image=photo)
        self.label.image = photo

        # Pack the label in the frame
        self.label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.label.bind("<Double-Button-1>", self.doubleleftClick)

        ########################################################################
        #
        # Back to Main Button

        # Create button to transition to back to main window
        mainbutton = ttk.Button(self, text="Main Menu",
                                command=lambda: controller.show_frame(MainMenuPage))

        # Pack button into frame
        mainbutton.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Select Colors Button

        # Create color button to transition to color select page
        colorbutton = ttk.Button(self, text="Select Colors",
                                command=lambda: controller.show_frame(SelectColorsPage))

        # Pack button into frame
        colorbutton.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Update Button

        # Create update button to update fractal
        updatebutton = ttk.Button(self, text="Update",
                                  command=lambda: updatepicture(self.label))

        # Pack button into frame
        updatebutton.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Color Slider Bar

        # Create color slider bar
        self.colorsliderbar = ttk.Scale(self, from_=0, to=255)

        self.colorsliderbar.grid(row=1, column=0, sticky='nsew')

        self.colorsliderbar.bind("<B1-Motion>", self.buttonhold)

    ############################################################################

    # Button release method for color slider bar
    def buttonrelease(self,event):

        global zdict, colorshift, colormultiplier, size

        RGBarray = zdict[2]
        escapenumbers = zdict[3]
        multiplier = zdict[4]

        value0 = np.mod(colormultiplier*escapenumbers + colorshift,256)
        [Rout, Gout, Bout] = colorfunction(value0,fourierprecision,ao,an,bn)
        
        R = (np.floor(Rout))*multiplier
        G = (np.floor(Gout))*multiplier
        B = (np.floor(Bout))*multiplier

        RGBarray[:,:,0] = R
        RGBarray[:,:,1] = G
        RGBarray[:,:,2] = B

        zdict[2] = RGBarray
        RGB = RGBarray
        img = Image.fromarray(RGB, 'RGB')
        photo = ImageTk.PhotoImage(img)

        self.label.config(image=photo)
        self.label.image = photo

    ############################################################################

    # Buttonhold method for colorslider
    def buttonhold(self,event):

        global colorshift

        colorshift = self.colorsliderbar.get()

        self.buttonrelease(event)

    ############################################################################

    # Double left click method (zooms in on image at location of double click)
    def doubleleftClick(self, event):

        # Global variables to be manipulated
        global xo, yo, x_offset, y_offset, count, zo, Z, zdict

        # Get coordinates of mouse click
        px = event.x
        py = event.y

        # Set new image center
        xo = xo + (2*(px/defaultsize) - 1)*x_offset
        yo = yo + (2*(py/defaultsize) - 1)*y_offset

        # Reset offset size
        x_offset = x_offset/zoom
        y_offset = x_offset

        # Reset arrays of x,y,z values
        x = np.linspace(xo-x_offset, xo+x_offset, defaultsize)
        y = np.linspace(yo-y_offset, yo+y_offset, defaultsize)
        X, Y = np.meshgrid(x,y)
        Z = X + 1j*Y

        # Reset zdict
        zdict[0] = Z
        zdict[1] = np.absolute(Z)
        zdict[2] = np.zeros([defaultsize,defaultsize,3], dtype=np.uint8)
        zdict[3] = np.zeros([defaultsize,defaultsize])
        zdict[4] = np.zeros([defaultsize,defaultsize])

        # Reset other variables
        count = 0
        zo = np.zeros([defaultsize,defaultsize])

        # Replot the image
        img = Image.fromarray(zdict[2], 'RGB')
        photo = ImageTk.PhotoImage(img)
        self.label.config(image=photo)
        self.label.image = photo
        

################################################################################
#
# Class for the Color Selector Page


class SelectColorsPage(tk.Frame):

    ############################################################################
    #
    # Methods of SelectColorsPage class

    # Init method - Automatically called when an instance of SelectColorsPage is made
    def __init__(self, parent, controller):

        ########################################################################
        #
        # Frame Initialization

        # Call init method from tk.Frame
        tk.Frame.__init__(self,parent)

        ########################################################################
        #
        # Main Label

        # Create label
        label = tk.Label(self, text="RGB Spectrum", font=LARGE_FONT)

        # Pack the label in the frame
        label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Main Menu Button

        # Create button to go back to the main menu
        mainbutton = ttk.Button(self, text="Main Menu",
                                command=lambda: controller.show_frame(MainMenuPage))

        # Pack button into frame
        mainbutton.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Fractal Window Button

        # Create button to go to the fractal window
        fracwinbutton = ttk.Button(self, text="Fractal Window",
                                command=lambda: controller.show_frame(FractalWindow))

        # Pack button into frame
        fracwinbutton.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        ########################################################################
        #
        # Plot of RGB values

        # Create a canvas for the figure
        canvas = FigureCanvasTkAgg(f, self)

        # Show the canvas
        canvas.show()

        # Specify canvas location on grid
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Add toolbar to canvas
        #toolbar = NavigationToolbar2TkAgg(canvas, self)
        #toolbar.update()
        #canvas._tkcanvas.pack()

        ########################################################################
        #
        # Current Entry Window

        # Create entry bar for Pixel x Pixel size of the fractal
        curentry = ttk.Entry(self)

        # Set default size value as defaultsize
        curentry.insert(10, 1)

        # Pack size enter window
        curentry.grid(row=4, column=2, columnspan=2, sticky="n", padx=10)

        ########################################################################
        #
        # Current Color Label

        # Create label
        label = tk.Label(self, text="Current Color to Modify", font=LARGE_FONT)

        # Pack the label in the frame
        label.grid(row=3, column=2, columnspan=2, sticky="s", padx=10)

################################################################################
#
# Main Function


def main():

    # Call FracWinApp class
    app = FracWinApp()

    # Animation Call
    ani = animation.FuncAnimation(f, animatecolorselection, interval=1000)

    # Put in mainloop so that the window does not close
    app.mainloop()
    
################################################################################
if __name__ == '__main__':
    main()
