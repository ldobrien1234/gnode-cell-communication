using DifferentialEquations
using Distributions
using Plots
using DelimitedFiles

#The number of training examples for a dataset
num_examples=20000

#-----------------------------------
#specify the signaling topology
#-----------------------------------

#Adjacency matrix for directed graph
topology = [0 1 1
            0 0 1
            0 0 0]

nCell = size(topology, 1) #3 cells in the topology

#-----------------------------------
#specify the number of iterations
#-----------------------------------

nits = 10

#-----------------------------------
#specify the value of A0
#-----------------------------------

A0 = 0.7

#-----------------------------------
#specify the value of lambda
#-----------------------------------

lambda = 1.0
kappa = 1.0

#-------------------------------------------------
#specify the wait time distribution and pulse time
#-------------------------------------------------

mean = 50.0

pulse = 5.0 # τ value in the paper

#-----------------------------------
# internal ODE system
#-----------------------------------

function GataPu1(du,u,p,t)
    a = p[1:5]
    b = p[6:8]
    g = p[9:17]
    L = p[18:20] #L = [A,B,C]
    du[1] = -b[1]*u[1] + (a[1]*L[1] + a[2]*u[1])/(1 + g[1]*L[1] + g[2]*u[1] + g[3]*u[1]*u[2])
    du[2] = -b[2]*u[2] + (a[3]*L[2] + a[4]*u[2])/(1 + g[4]*L[2] + g[5]*u[2] + g[6]*u[1]*u[2] + g[7]*u[1]*u[3])
    du[3] = -b[3]*u[3] + (a[5]*u[1])/(1 + g[8]*u[1] + g[9]*L[3])
end

#-----------------------------------
# parameter vector p
#-----------------------------------

p = zeros(0)
a = [1.0,0.25,1.0,0.25,0.01]
b = [0.01,0.01,0.01]
g = [1.0,0.25,1.0,1.0,0.25,1.0,0.13,0.01,10]
L = [1.0,0.5,0] #L = [A,B,C] = [A,0.5,0]
append!(p,a)
append!(p,b)
append!(p,g)
append!(p,L)

#-----------------------------------
# wait time distribution
#-----------------------------------

function rand_exponential(mean)
    if mean <= 0.0
        error("mean must be positive")
    end
    -mean*log(rand())
end

Nml = Normal(50.0, 10.0)


#to take discrete time points to update A from a random distribution
function wait_time(mean)
    round(rand_exponential(mean))
end

#initialize dataset for storage
initial_features = Any[]
final_features = Any[]

for example=1:num_examples
    #-----------------------------------
    #Simulation Code
    #-----------------------------------
    #Initializing values
    global CellValues = [[rand(Uniform(0,40)),rand(Uniform(0,10)),rand(Uniform(0,40)), A0] for i=1:nCell, n=1:1]

    #getting the initial feature matrix for the dataset
    global InitialValues = deepcopy(CellValues) #CellValues is modified later
    push!(initial_features,InitialValues)

    global CellIts = zeros(nCell)
    #Taking discrete time points to update A
    #3 time points b/c there are three cells?
    global CellWaitTimes = [wait_time(mean) for i=1:nCell, n=1:1]
    global CellState = ["wait" for i=1:nCell, n=1:1]
    global PulseCoeffs = zeros(Float64, nCell, nCell)

    ### Define empty vectors for G,P,X concentrations and times for each cell
    for x in [Symbol("Cell"*string(i)) for i=1:nCell]
        @eval $x = Any[] #Any is a data type; it includes all other data types
    end
    for x in [Symbol("Time"*string(i)) for i=1:nCell]
        @eval $x = Any[]
    end

    global total_time = 0.0

    #Number of iterations of solver
    while min(CellIts...) < 1000 + 40*nCell
        #taking the smallest cell from CellWaitTimes as the end of
        #must stop at this time to update one of the A parameters
        t = min(CellWaitTimes...) #time for the integration interval

        #loop to update all the cells (using the ODEs)
        for i = 1:nCell
            u0 = [CellValues[i][1], CellValues[i][2], CellValues[i][3]]
            p[18] = CellValues[i][4]
            tspan = (0.0, t)
            prob = ODEProblem(GataPu1,u0,tspan,p)
            sol = solve(prob, save_everystep=false)
            #storing the data
            append!(eval(Symbol("Cell"*string(i))), sol.u)
            append!(eval(Symbol("Time"*string(i))), sol.t .+ total_time)


            CellValues[i][1] = sol[length(sol)][1]
            CellValues[i][2] = sol[length(sol)][2]
            CellValues[i][3] = sol[length(sol)][3]
        end

        global CellWaitTimes = CellWaitTimes .- t # .- means elementwise subtraction
        global total_time = total_time + t

        #Update A values (PulseCoeffs) based on communication topology
        #for loop iterates through the cells
        for i = 1:nCell
            #only update A (PulseCoeffs) if signal wait time is zero
            if CellWaitTimes[i] == 0
                #all cell states start as "wait"
                if CellState[i] == "wait"
                    for m = 1:nCell
                        if topology[i,m] == 1
                            PulseCoeffs[i,m] = (CellValues[i][1]/(lambda + CellValues[i][2]))
                            CellValues[m][4] = CellValues[m][4] * PulseCoeffs[i,m]
                        elseif topology[i,m] == -1
                            PulseCoeffs[i,m] = (CellValues[i][2]/(kappa + CellValues[i][1]))
                            CellValues[m][4] = CellValues[m][4] * PulseCoeffs[i,m]
                        end
                    end
                    CellWaitTimes[i] = pulse
                    CellState[i] = "pulse"
                    CellIts[i] = CellIts[1] + 1
                else
                    for m = 1:nCell
                        if topology[i,m] == 1
                            CellValues[m][4] = CellValues[m][4] / PulseCoeffs[i,m]
                        elseif topology[i,m] == -1
                            CellValues[m][4] = CellValues[m][4] / PulseCoeffs[i,m]
                        end
                    end

                    CellWaitTimes[i] = wait_time(mean)
                    CellState[i] = "wait"
                end
            end
        end
    end

    #getting the final state of each cell
    push!(final_features, CellValues)
print(example)
end

#Writing the final data to a delimited file
open("cell_dataset_20000.txt", "w") do io
    writedlm(io, [initial_features, final_features])
end
