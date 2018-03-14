################################################################################
#
# AUTONOMOUS PATH PLANNING
# Written by Toren Wallengren
#
# This script generates random obstacles on the plane and then finds paths
# between chosen start and end coordinates that avoid those obstacles
#
# This is done by taking the differential equations derived in the video,
# subtracting the right hand side of the equation from both sides (to get
# x''(t) - F(x,y,x',y',t) = 0 and y''(t) - G(x,y,x',y',t) = 0) and then using
# Newton's method to find the zeros of the resulting left hand side (the left
# hand side equals zero by definition, so we've reformulated a differential
# equations problem as a root-finding problem)
#
# Note this is discretized - they were derived in a 'continuous' way, but must
# be discretized to be solved in a computer. I won't go too deep into exactly
# how this was done here but it is good to keep in mind.
#
################################################################################
################################################################################
# Import necessary libraries
import math, random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.functions import exp

################################################################################
# Function being solved by Newton's Method in BVP solver function

def newtonsfunction(F,G,x0,x1,x2,y0,y1,y2,h):
    # F is the right hand side of the differential equation for x''(t)
    # G is the right hand side of the differential equation for y''(t)
    # (x0,y0);(x1,y1);(x2,y2) are three consecutive points of the discretized
    # 'guess' solution (x,y)
    # h is the step size in time between the consecutive points

    # This is the expression derived in the video. If something is a solution
    # to the derived ODE's, then plugging in points from that solution to this
    # expression will yield values extremely close to 0 (and should approach 0
    # as h approaches 0). So if we plug a guess in that is not a solution,
    # these expressions can provide a 'measurement' of how far off it is from
    # an actual solution.
    xout = (x2 - 2*x1 + x0)/h - F(x1,y1,(x2-x0)/(2*h),(y2-y0)/(2*h))
    yout = (y2 - 2*y1 + y0)/h - G(x1,y1,(x2-x0)/(2*h),(y2-y0)/(2*h))

    return xout, yout

################################################################################
# Function to populate & invert Jacobian matrix - this is the 'meat' of the
# script. Unfortunately, it is also a part of the script that cannot be fully
# explained through comments. The best way to think of this is to use an
# analogy with simpler functions. If we define f(x) = 3x^2 - 2x - 7, this part
# of the script would be functionally equivalent to computing 1/f'(x). But in
# this script, the output of f(x) is a vector and the input x is also a vector.

def jacobinv(F,G,Fx,Gx,Fy,Gy,Fxp,Gxp,Fyp,Gyp,ALPHA,BETA,solmat,N,h):
    # F is the right hand side of the x-component of the ODE
    # G is the right hand side of the y-component of the ODE
    # The small x, y, xp, or yp next to the other F's and G's in the input
    # indicate that it is the derivative of F or G with respect to that variable
    # ALPHA is the starting coordinate
    # BETA is the ending coordinate
    # solmat is a vector with 2*N components which is our 'guess' solution. The
    #   first N components represent the x-coordinates and the second N
    #   represent the y-coordinates
    # N is the number of coordinates points in our 'guess' solution
    # h is the time step

    # Initialize matrix
    jac = np.zeros([2*N,2*N])

    # Populate the first row of M11 (upper left square of jac)
    jac[0,0] = -2/h**2 - Fx(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
    jac[0,1] = 1/h**2 - (1/(2*h))*Fxp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

    # Populate the first row of M21 (lower left square of jac)
    jac[N,0] = -Gx(solmat[0],solmat[N],
                    (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
    jac[N,1] = (-1/(2*h))*Gxp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

    # Populate the first row of M12 (upper right square of jac)
    jac[0,N] = -Fy(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
    jac[0,N+1] = (-1/(2*h))*Fyp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

    # Populate the first row of M22 (lower right square of jac)
    jac[N,N] = -2/h**2 - Gy(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
    jac[N,N+1] = 1/h**2 - (1/(2*h))*Gyp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

    # Loop to populate intermediate values of jac
    for i in range(1,N-1):

        # Populate intermediate values of M11
        jac[i,i-1] = 1/h**2 + (1/(2*h))*Fxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[i,i] = -2/h**2 - Fx(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[i,i+1] = 1/h**2 - (1/(2*h))*Fxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

        # Populate intermediate values of M21
        jac[N+i,i-1] = (1/(2*h))*Gxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[N+i,i] = -Gx(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[N+i,i+1] = (-1/(2*h))*Gxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

        # Populate intermediate values of M12
        jac[i,N+i-1] = (1/(2*h))*Fyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[i,N+i] = -Fy(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[i,N+i+1] = (-1/(2*h))*Fyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

        # Populate intermediate values of M22
        jac[N+i,N+i-1] = 1/h**2 + (1/(2*h))*Gyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[N+i,N+i] = -2/h**2 - Gy(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
        jac[N+i,N+i+1] = 1/h**2 - (1/(2*h))*Gyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

    # Populate final values of M11
    jac[N-1,N-2] = 1/h**2 + (1/(2*h))*Fxp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
    jac[N-1,N-1] = -2/h**2 - Fx(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

    # Populate final values of M21
    jac[2*N-1,N-2] = (1/(2*h))*Gxp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
    jac[2*N-1,N-1] = -Gx(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

    # Populate final values of M12
    jac[N-1,2*N-2] = (1/(2*h))*Fyp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
    jac[N-1,2*N-1] = -Fy(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

    # Populate final values of M22
    jac[2*N-1,2*N-2] = 1/h**2 + (1/(2*h))*Gyp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
    jac[2*N-1,2*N-1] = -2/h**2 - Gy(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

    jacinv = np.linalg.inv(jac)

    return jacinv

################################################################################
# Boundary Value Problem Solver Function
# ALPHA - Starting location
# BETA - Ending location
# F - Symbolic function for x''
# G - Symbolic function for y''
# P0x & P0y - Symbolic functions for initial guess - Must be parametrized by 'ts'
#       and satisfy P0(0) = ALPHA & P0(1) = BETA
# MAXITER - Max number of iterations for solver
# TOL - Precision tolerance for solver
# N - Number of discrete intervals

def pathplanningbvpsolver(ALPHA,BETA,F,G,P0x,P0y,MAXITER,TOL,N):

    # Take necessary derivatives of F and G
    # F & G must depend on xs, ys, xps, and yps
    Fx = sp.diff(F,xs)
    Gx = sp.diff(G,xs)
    Fxp = sp.diff(F,xps)
    Gxp = sp.diff(G,xps)
    Fy = sp.diff(F,ys)
    Gy = sp.diff(G,ys)
    Fyp = sp.diff(F,yps)
    Gyp = sp.diff(G,yps)

    # "Lambdify" all symbolic functions
    # Allows calls to evaluate functions at specific points
    F = sp.lambdify((xs,ys,xps,yps),F)
    G = sp.lambdify((xs,ys,xps,yps),G)
    Fx = sp.lambdify((xs,ys,xps,yps),Fx)
    Gx = sp.lambdify((xs,ys,xps,yps),Gx)
    Fxp = sp.lambdify((xs,ys,xps,yps),Fxp)
    Gxp = sp.lambdify((xs,ys,xps,yps),Gxp)
    Fy = sp.lambdify((xs,ys,xps,yps),Fy)
    Gy = sp.lambdify((xs,ys,xps,yps),Gy)
    Fyp = sp.lambdify((xs,ys,xps,yps),Fyp)
    Gyp = sp.lambdify((xs,ys,xps,yps),Gyp)
    P0x = sp.lambdify(ts,P0x)
    P0y = sp.lambdify(ts,P0y)

    # Initialize 't' to build guess function
    t = np.linspace(0,1,N+2)

    # Calculate increment value (h) for use in equations
    h = t[1] # t[1] - t[0] = t[1] because t[0] is always 0

    # Initialize solution arrays (need 2 to compute error)
    solprev = np.zeros([2*N,1])
    solnew = np.zeros([2*N,1])

    # Loop to populate values of solution array (solprev)
    for i in range(0,N):

        solprev[i] = P0x(t[i+1])
        solprev[N+i] = P0y(t[i+1])

    # Initialize array for Newtons Function
    NF = np.zeros([2*N,1])

    # Initialize counter
    count = 0

    # Initialize error
    error = 1

    while ((error >= TOL) & (count < MAXITER)):

        # Calculate first values of Newtons Function
        NF[0], NF[N] = newtonsfunction(F,G,ALPHA[0],solprev[0],solprev[1],ALPHA[1],
                                 solprev[N],solprev[N+1],h)

        # Loop to calculate intermediate values of NF
        for i in range(1,N-1):
            NF[i], NF[N+i] = newtonsfunction(F,G,solprev[i-1],solprev[i],solprev[i+1],
                                         solprev[N-1+i],solprev[N+i],solprev[N+1+i],h)

        # Calculate end values of NF
        NF[N-1], NF[2*N-1] = newtonsfunction(F,G,solprev[N-2],solprev[N-1],BETA[0],
                                         solprev[2*N-2],solprev[2*N-1],BETA[1],h)

        jacinv = jacobinv(F,G,Fx,Gx,Fy,Gy,Fxp,Gxp,Fyp,Gyp,ALPHA,BETA,solprev,N,h)

        # This next line is where newton's method is actually implemented
        solnew = solprev - np.dot(jacinv,NF)

        # Find the relative error between the old and new solution
        errorvec = solnew - solprev
        error = np.linalg.norm(errorvec)

        solprev = solnew

        count += 1

        # I print here to see the progress of the script as it goes. It is
        # non-essential, feel free to delete it if desired.
        print(count,error)
        
    return solnew
################################################################################
# Initialize symbolic variables - using symbolic expressions will allow us to
# take the derivative of expressions within the code.

xs = sp.Symbol('xs') # 'xs' stands for x-symbolic
ys = sp.Symbol('ys') # 'ys' stands for y-symbolic
xps = sp.Symbol('xps') # 'xps' stands for xprime-symbolic
yps = sp.Symbol('yps') # 'yps' stands for yprime-symbolic
ts = sp.Symbol('ts') # 'ts' stands for t-symbolic (time)


################################################################################
# Initialize numeric variables
# Note you'll start to see 'rov' being used throughout the rest of the script.
# I had originally worked on this as a means to make a rover navigate between
# two points on a plane with obstacles between. I'm too lazy to take this
# reference out and have decided to type comments about it instead, which
# has probably taken just as much time to do as it would to change 'rov', but
# I've already committed to this choice and don't want to double the time spent.

rov = np.array([-2,-2]) # Starting position of rover
target = np.array([12,12]) # Final position of rover
xinterval = np.array([0,10]) # Range of x-coordinates for obstacles
yinterval = np.array([0,10]) # Range of y-coordinates for obstacles
M = 6 # Number of obstacles
obstaclesx = (xinterval[0] + (xinterval[1] - xinterval[0])
              *np.random.rand(M,1)) # x-coordinates of obstacles
obstaclesy = (yinterval[0] + (yinterval[1] - yinterval[0])
              *np.random.rand(M,1)) # y-coordinates of obstacles


################################################################################
# Initialize cost function & its derivatives

cost = 1 # First term in cost function

# Loop to add terms to cost function
for i in range(0,M):

    xm = obstaclesx[i] # x-coordinate of current obstacle
    ym = obstaclesy[i] # y-coordinate of current obstacle
    termtoadd = sp.exp(-((-xs + xm)**2 + (-ys + ym)**2)) # Term to add on each iteration
    cost = cost + termtoadd # Add term to cost function

costx = sp.diff(cost,xs) # First x-derivative of cost
costy = sp.diff(cost,ys) # First y-derivative of cost

################################################################################
# Initialize ODE system from Autonomous Path Planning Video (alpha/beta = u/v)

denom = 2*cost # Denominator for ODE's (to simplify syntax)

f = (costx*(yps**2 - xps**2) - 2*costy*xps*yps)/denom # x-double prime
g = (costy*(xps**2 - yps**2) - 2*costx*xps*yps)/denom # y-double prime

# Number of discretizations
N = 25

# Max iterations
MAXITER = 1000

# Tolerance
TOL = 0.01

# Plot start point, end point, and obstacles
plt.plot(rov[0],rov[1],'ro')
plt.plot(target[0],target[1],'ro')
plt.plot(obstaclesx,obstaclesy,'x')

# Initial guess path - a straight line between the points
P0x = (target[0] - rov[0])*ts + rov[0]
P0y = (target[1] - rov[1])*ts + rov[1]

# Find the real solution using the guess
sol = pathplanningbvpsolver(rov,target,f,g,P0x,P0y,MAXITER,TOL,N)

# Add the start and end coordinates to the solution
solx = np.zeros([N+1,1])
solx[0] = rov[0]
solx[1:N] = sol[0:N-1]
solx[N] = target[0]

soly = np.zeros([N+1,1])
soly[0] = rov[1]
soly[1:N] = sol[N:2*N-1]
soly[N] = target[1]

# Plot solution
plt.plot(solx,soly)

# Repeat the above process for guess inputs that are curved
# Feel free to change the range here to generate more solutions
# Just keep in mind that solutions can overlap, so it is not expected to see
# a unique output path for each guess input
for i in range(2,3):

    # Another thing I print to gauge the progress of the code, feel free to
    # delete if desired
    print("guess number: ", i)

    # Initial guess path - for each i, the path will be a curve that looks like
    # the function f(x) = x^i from x=0 to x=1
    P01x = (target[0] - rov[0])*ts**i + rov[0]
    P01y = (target[1] - rov[1])*ts + rov[1]

    # Initialize another guess path - this one will look like the function
    # f(x) = x^(1/i) from x=0 to x=1, also can be thought of as the reflection
    # of f(x) = x^i across the line y=y(x)=x.
    P02x = (target[0] - rov[0])*ts + rov[0]
    P02y = (target[1] - rov[1])*ts**i + rov[1]

    sol1 = pathplanningbvpsolver(rov,target,f,g,P01x,P01y,MAXITER,TOL,N)
    sol2 = pathplanningbvpsolver(rov,target,f,g,P02x,P02y,MAXITER,TOL,N)

    sol1x = np.zeros([N+1,1])
    sol1x[0] = rov[0]
    sol1x[1:N] = sol1[0:N-1]
    sol1x[N] = target[0]

    sol1y = np.zeros([N+1,1])
    sol1y[0] = rov[1]
    sol1y[1:N] = sol1[N:2*N-1]
    sol1y[N] = target[1]

    sol2x = np.zeros([N+1,1])
    sol2x[0] = rov[0]
    sol2x[1:N] = sol2[0:N-1]
    sol2x[N] = target[0]

    sol2y = np.zeros([N+1,1])
    sol2y[0] = rov[1]
    sol2y[1:N] = sol2[N:2*N-1]
    sol2y[N] = target[1]

    plt.plot(sol1x,sol1y)
    plt.plot(sol2x,sol2y)

# Define limits of the plot, then show the plot
plt.xlim([-4,14])
plt.ylim([-4,14])
plt.show()
