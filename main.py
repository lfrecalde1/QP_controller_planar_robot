import numpy as np
import time
import mujoco
import mujoco.viewer
from fancy_plots import fancy_plots_2, fancy_plots_1
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import osqp
import scipy as sp
from scipy import sparse


def get_system_states(system):
    # System values angular displacement
    q1 = system.qpos[0]
    q2 = system.qpos[1]

    # System values angular velocities
    q1p = system.qvel[0]
    q2p = system.qvel[1]

    x = np.array([q1, q2, q1p, q2p], dtype=np.double)
    return x
def get_system_states_sensor(system):
    q0 = system.sensor("q_0").data.copy()
    q1 = system.sensor("q_1").data.copy()
    q0p = system.sensor("q_0p").data.copy()
    q1p = system.sensor("q_1p").data.copy()
    x = np.array([q0[0], q1[0], q0p[0], q1p[0]], dtype=np.double)
    return x

def get_system_energy(system):
    e1 = system.energy[0]
    e2 = system.energy[1]
    et = e1 + e1
    return et

def control_action(system, u):
    u1 = u[0]
    u2 = u[1]
    system.ctrl[1] = u1
    system.ctrl[3] = u2
    return None

def forward_kinematics(L, q):
    # Split Values of the system
    q_1 = q[0]
    q_2 = q[1]
    q_1p = q[2]
    q_2p = q[3]

    l1 = L[0]
    l2 = L[1]

    x = l1*np.cos(q_1) + l2*np.cos(q_1 + q_2)
    y = l1*np.sin(q_1) + l2*np.sin(q_1 + q_2)

    h = np.array([x, y], dtype=np.double)
    return h

def Jacobian_system(L, q):
    # Split Values of the system
    q_1 = q[0]
    q_2 = q[1]
    q_1p = q[2]
    q_2p = q[3]

    l1 = L[0]
    l2 = L[1]

    # Jacobian Matrix control of the system
    J11 = -l1*np.sin(q_1)-l2*np.sin(q_1 + q_2)
    J12 = -l2*np.sin(q_1 + q_2)
    J21 = l1*np.cos(q_1)+l2*np.cos(q_1 + q_2)
    J22 = l2*np.cos(q_1 + q_2)

    J = np.array([[J11, J12],[J21, J22]], dtype=np.double)
    return J

def get_hessian_1(L, q):
    # Split Values of the system
    q1 = q[0]
    q2 = q[1]
    q1p = q[2]
    q2p = q[3]

    l1 = L[0]
    l2 = L[1]
    # Jacobian Matrix control of the system
    H11 = -l2*np.sin(q1 + q2) - l1*np.sin(q1)
    H12 = -l2*np.sin(q1 + q2)
    H21 = l2*np.cos(q1 + q2) + l1*np.cos(q1)
    H22 = l2*np.cos(q1 + q2)

    H = np.array([[H11, H12],[H21, H22]], dtype=np.double)
    return H

def get_hessian_2(L, q):
    # Split Values of the system
    q1 = q[0]
    q2 = q[1]
    q1p = q[2]
    q2p = q[3]

    l1 = L[0]
    l2 = L[1]
    # Jacobian Matrix control of the system
    H11 = -l2*np.sin(q1 + q2)
    H12 = -l2*np.sin(q1 + q2)
    H21 = l2*np.cos(q1 + q2)
    H22 = l2*np.cos(q1 + q2)

    H = np.array([[H11, H12],[H21, H22]], dtype=np.double)
    return H

def get_manipulability(L, q):
    # Get Jacobian of the system
    J = Jacobian_system(L, q)
    J_t = J.transpose()

    aux_product = J@J_t
    determinante = np.linalg.det(aux_product)
    m = np.sqrt(determinante)
    return m

def get_manipulability_filter(L, q):
    # Get Jacobian of the system
    J = Jacobian_system(L, q)
    J_t = J.transpose()

    aux_product = J@J_t
    determinante = np.linalg.det(aux_product)
    m = np.sqrt(determinante)
    a_m = 0.1
    aux_m = np.exp(-((m)**2)/a_m)
    return aux_m

def get_jacobian_manipulability(L, q):
    # get Hessians of the system
    hessian_1 = get_hessian_1(L, q)
    hessian_2 = get_hessian_2(L, q)


    # Get Jacobian of the system         
    J = Jacobian_system(L, q)
    J_t = J.transpose()

    # Auxiliar values 
    aux_hessian_1 = J@hessian_1.transpose()
    aux_hessian_2 = J@hessian_2.transpose()
    aux_J = np.linalg.inv(J@J_t);
            
    vec_h_1 = aux_hessian_1.reshape(4, 1)
    vec_h_2 = aux_hessian_2.reshape(4, 1)
    vec_J = aux_J.reshape(4, 1)

    # Manipulability of the system 
    mani = get_manipulability(L, q)

    # Creation Manipulability matrix
    J_11 = mani*(vec_h_1.transpose()@vec_J)
    J_21 = mani*(vec_h_2.transpose()@vec_J)

    # Creation auxiliar matrix of the system 
    J_m = np.array([[J_11[0,0]],[J_21[0,0]]], dtype=np.double)

    return J_m
def get_jacobian_manipulability_filter(L, q):
    J_m = get_jacobian_manipulability(L, q)
    aux_v = get_manipulability_filter(L, q)
    aux_m = get_manipulability(L, q)
    aux = aux_v*-(2*(aux_m)/0.1)
    J_mf = aux*J_m.transpose()
    return J_mf
            
def control_law(hd, hdp, h, q, L):
    # get system Jacobian
    J = Jacobian_system(L, q)
    # Control Error
    e = hd-h
    he = e.reshape((2, 1))
    hdp = hdp.reshape((2,1))
    
    u = np.linalg.inv(J)@(1*np.tanh(he))
    return u[:,0]
def QP_controller_complete(hd, h, q_n, L, hdp):
    n = 3
    m = 2

    J_m_sin = get_jacobian_manipulability(L, q_n)
    q = np.array([0, 0, 0, 0, 0])
    mani = get_manipulability(L, q_n)

    Pn = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Pm = sparse.csc_matrix([[1, 0], [0, 1]])
    P = sparse.block_diag([Pm, Pn], format='csc')
    J = Jacobian_system(L, q_n)
    J_m = get_jacobian_manipulability(L, q_n)
    Ad = sparse.csc_matrix([[J[0, 0], J[0, 1]], [J[1, 0], J[1, 1]], [J_m[0, 0], J_m[1, 0]]])
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(n)]),
        sparse.hstack([sparse.eye(m), sparse.csc_matrix((m, n))*0])], format='csc')

    he = 1*np.tanh((hd - h))
    me = 1*(1 - mani)

    l = np.hstack([he, me, -2*np.ones(m)])
    u = np.hstack([he, me,  2*np.ones(m)])
    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u)

    return prob
def QP_controller(hd, h, q_n, L, hdp):
    n = 2
    m = 2

    q = np.array([0, 0, 0, 0])

    Pn = sparse.csc_matrix([[1, 0], [0, 1]])
    Pm = sparse.csc_matrix([[1, 0], [0, 1]])
    P = sparse.block_diag([Pm, Pn], format='csc')
    J = Jacobian_system(L, q_n)
    Ad = sparse.csc_matrix([[J[0, 0], J[0, 1]], [J[1, 0], J[1, 1]]])
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(n)]),
        sparse.hstack([sparse.eye(m), sparse.csc_matrix((m, n))*0])], format='csc')

    he = 1*np.tanh((hd - h)) 

    l = np.hstack([he, -2*np.ones(m)])
    u = np.hstack([he,  2*np.ones(m)])
    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u)

    return prob
def QP_solver_complete(prob, hd, h, q_n, L,  hdp):
    n = 3
    m = 2

    q_new = np.array([0, 0, 0, 0, 0])
    mani = get_manipulability(L, q_n)
    mani_filter = get_manipulability_filter(L, q_n)

    Pn = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Pm = sparse.csc_matrix([[1, 0], [0, 1]])
    P = sparse.block_diag([Pm, Pn], format='csc')
    J = Jacobian_system(L, q_n)
    J_m = get_jacobian_manipulability(L, q_n)
    Ad = sparse.csc_matrix([[J[0, 0], J[0, 1]], [J[1, 0], J[1, 1]], [J_m[0, 0], J_m[1, 0]]])
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(n)]),
        sparse.hstack([sparse.eye(m), sparse.csc_matrix((m, n))*0])], format='csc')

    he = np.abs(1-mani_filter)*np.tanh((hd - h)) + hdp
    me = 1*mani_filter*(1 - mani)

    l_new = np.hstack([he, me, -2*np.ones(m)])
    u_new = np.hstack([he, me,  2*np.ones(m)])
    prob.update(Px=P.data, q = q_new,  Ax= A.data, l = l_new, u = u_new)
    res = prob.solve()

    return res.x[0:2]


def QP_solver(prob, hd, h, q_n, L,  hdp):
    n = 2
    m = 2

    q_new = np.array([0, 0, 0, 0])

    Pn = sparse.csc_matrix([[10, 0], [0, 10]])
    Pm = sparse.csc_matrix([[1, 0], [0, 1]])
    P = sparse.block_diag([Pm, Pn], format='csc')
    J = Jacobian_system(L, q_n)
    Ad = sparse.csc_matrix([[J[0, 0], J[0, 1]], [J[1, 0], J[1, 1]]])
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(n)]),
        sparse.hstack([sparse.eye(m), sparse.csc_matrix((m, n))*0])], format='csc')

    he = 1*np.tanh((hd - h))

    l_new = np.hstack([he, -2*np.ones(m)])
    u_new = np.hstack([he, 2*np.ones(m)])
    prob.update(Px=P.data, q = q_new,  Ax= A.data, l = l_new, u = u_new)
    res = prob.solve()

    return res.x[0:2]

def main():
    # Load Model form XML file
    m = mujoco.MjModel.from_xml_path('prueba_1.xml')
    # Get information form the xml
    data = mujoco.MjData(m)

    # System Variables
    l1 = 1
    l2 = 1
    L = [l1, l2]

    # Simulation time parameters
    ts = 0.01
    tf = 40
    t = np.arange(0, tf+ts, ts, dtype=np.double)

    # States System
    q = np.zeros((4, t.shape[0]+1), dtype=np.double)
    q_n = np.zeros((4, t.shape[0]+1), dtype=np.double)

    # Vector Forward Kineamtics
    h = np.zeros((2, t.shape[0]+1), dtype=np.double)
    mani = np.zeros((2, t.shape[0]+1), dtype=np.double)

    # Desired Trajectory of the sytem
    hd = np.zeros((2, t.shape[0]), dtype=np.double)
    hdp = np.zeros((2, t.shape[0]), dtype=np.double)
    hd[0, :] = 1
    hd[1, :] = 1

    hdp[0, :] = 0
    hdp[1, :] = 0
    # Empty control error
    he = np.zeros((2, t.shape[0]), dtype=np.double)

    # Control signals
    u = np.zeros((2, t.shape[0]), dtype=np.double)

    # States Energy system
    E = np.zeros((1, t.shape[0]+1), dtype=np.double)

    # Define Paramerters for the software
    m.opt.timestep = ts

    # Reset Properties system
    mujoco.mj_resetDataKeyframe(m, data, 0)  # Reset the state to keyframe 0

    # Initial conditions system
    data.qpos[0] = 180*np.pi/180
    data.qpos[1] = -1*np.pi/180


    with mujoco.viewer.launch_passive(m, data) as viewer:
        if viewer.is_running():
            # Initial data System
            q[:, 0] = get_system_states(data)
            # Initial Energy System
            E[:, 0] = get_system_energy(data)
            q_n[:, 0] = get_system_states_sensor(data)

            # Forward Kinematics
            h[:, 0] = forward_kinematics(L, q_n[:, 0])
            mani[:, 0] = get_manipulability(L, q_n[:, 0])

            QP = QP_controller(hd[:, 0], h[:, 0], q_n[:, 0], L, hdp[:, 0])

            # Simulation of the system
            for k in range(0, t.shape[0]):
                tic = time.time()
                if t[k] >=20:
                    hd[0, k] = 3
                    hd[1, k] = 3
                else:
                    None
                # Control Vector
                he[:, k] = hd[:, k] - h[:, k]
                

                # Section to control the system
                u[:, k] = control_law(hd[:, k], hdp[:, k], h[:, k], q_n[:, k], L)
                #opti = QP_solver(QP, hd[:, k], h[:, k], q_n[:, k], L, hdp[:, k])
                #u[:, k] = opti
                control_action(data, u[:,k])

                # System evolution
                mujoco.mj_step(m, data)

                # System evolution visualization
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # Get system states
                q[:, k+1] = get_system_states(data)
                E[: k+1] = get_system_energy(data)
                q_n[:, k+1] = get_system_states_sensor(data)
                h[:, k+1] = forward_kinematics(L, q_n[:, k+1])
                mani[:, k+1] = get_manipulability(L, q_n[:, k+1])


                # Section to guarantee same sample times
                while (time.time() - tic <= m.opt.timestep):
                    None
                toc = time.time() - tic

        fig2, ax12, ax22 = fancy_plots_2()
        ## Axis definition necesary to fancy plots
        ax12.set_xlim((t[0], t[-1]))
        ax22.set_xlim((t[0], t[-1]))
        ax12.set_xticklabels([])

        state_q1p_d, = ax12.plot(t,u[0,0:t.shape[0]],
                    color='#00429d', lw=2, ls="-")
        state_q1p, = ax12.plot(t,q_n[2,0:t.shape[0]],
                    color='#9e4941', lw=2, ls="-.")

        state_q2p_d, = ax22.plot(t,u[1,0:t.shape[0]],
                    color='#ac7518', lw=2, ls="-")
        state_q2p, = ax22.plot(t,q_n[3,0:t.shape[0]],
                    color='#97a800', lw=2, ls="-.")

        ax12.set_ylabel(r"$[rad/s]$", rotation='vertical')
        ax12.legend([state_q1p_d,state_q1p],
            [r'$\dot{{q}}^c_1$', r'$\dot{{q}}_1$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

        ax22.set_ylabel(r"$[rad/s]$", rotation='vertical')
        ax22.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
        ax22.legend([state_q2p_d, state_q2p],
            [r'$\dot{{q}}^c_2$', r'$\dot{{q}}_2$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
        ax22.grid(color='#949494', linestyle='-.', linewidth=0.5)
        fig2.savefig("system_states_noise.eps")
        fig2.savefig("system_states_noise.png")
        fig2
        plt.show()

        fig1, ax11 = fancy_plots_1()
        ## Axis definition necesary to fancy plots
        ax11.set_xlim((t[0], t[-1]))

        error_x, = ax11.plot(t[0:he.shape[1]],he[0,:],
                        color='#BB5651', lw=2, ls="-")
        error_y, = ax11.plot(t[0:he.shape[1]],he[1,:],
                        color='#00429d', lw=2, ls="-")

        ax11.set_ylabel(r"$\textrm{Control Error}[m]$", rotation='vertical')
        ax11.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
        ax11.legend([error_x, error_y],
                [r'$^i \tilde{x}_b$', r'$^i \tilde{y}_b$'],
                loc="best",
                frameon=True, fancybox=True, shadow=False, ncol=2,
                borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
                borderaxespad=0.3, columnspacing=2)
        ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

        fig1.savefig("control_error.eps")
        fig1.savefig("control_error.png")
        fig1
        plt.show()

        fig3, ax13 = fancy_plots_1()
        ## Axis definition necesary to fancy plots
        ax13.set_xlim((t[0], t[-1]))

        manipulability, = ax13.plot(t,mani[0,0:t.shape[0]],
                        color='#BB5651', lw=2, ls="-")

        ax13.set_ylabel(r"$\textrm{Manipulability}[m]$", rotation='vertical')
        ax13.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
        ax13.legend([manipulability],
                [r'$m$'],
                loc="best",
                frameon=True, fancybox=True, shadow=False, ncol=2,
                borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
                borderaxespad=0.3, columnspacing=2)
        ax13.grid(color='#949494', linestyle='-.', linewidth=0.5)

        fig3.savefig("manipulability.eps")
        fig3.savefig("manipulability.png")
        fig3
        plt.show()
        
if __name__ == '__main__':
    try:
        main()
    except(KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass