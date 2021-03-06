using DifferentialEquations
using Distributions
using Plots
using DelimitedFiles

#number of cells
nCell = 3
#The number of training examples for a dataset
num_examples = 50000
#The number of evaluations for time-series data
num_eval = 100
#the end of the numerical integration interval
t_f = 1600

eval_interval = t_f / num_eval

#A0 constant used for the ODEs
A0 = 0.7

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
L = [A0,0.5,0] #L = [A,B,C] = [A,0.5,0]
append!(p,a)
append!(p,b)
append!(p,g)
append!(p,L)

#initialize dataset for storage
initial_features = Any[]
final_features = Any[]

for i=1:num_examples
    print(i)
    #-----------------------------------
    #Simulation Code
    #-----------------------------------
    #Initializing values
    global CellValues = [rand(Uniform(0,40)),rand(Uniform(0,10)),rand(Uniform(0,40)), A0]

    #getting the initial feature matrix for the dataset
    global InitialValues = deepcopy(CellValues) #CellValues is modified later
    push!(initial_features, InitialValues)

    #loop to update all the cells (using the ODEs)

    u0 = [CellValues[1], CellValues[2], CellValues[3]]
    p[18] = CellValues[4]
    tspan = (0.0, t_f)
    prob = ODEProblem(GataPu1,u0,tspan,p)
    sol = solve(prob)

    global Targets = Any[]
    for m=1:num_eval
        G1 = sol(m*eval_interval)[1]
        P1 = sol(m*eval_interval)[2]
        X1 = sol(m*eval_interval)[3]
        #CellValues[i][2] = [[sol(m*eval_interval)[2]] for m=1:num_eval, n=1:1]
        #CellValues[i][3] = [[sol(m*eval_interval)[3]] for m=1:num_eval, m=1:1]
        push!(Targets, [G1,P1,X1])
    end

#getting the final state of each cell
push!(final_features, [Targets])
end

#Writing the final data to two delimited files
open("Cell1_Features.txt", "w") do io
    writedlm(io, initial_features)
end

open("Cell1_TimeSeries.txt", "w") do io
    writedlm(io, final_features)
end
