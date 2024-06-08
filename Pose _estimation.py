import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import warnings

warnings.filterwarnings("ignore")

def getTrack(vid_path : str, disp_vid = 'True', disp_plot = 'True'):  # Define a function to track corners in a video, with parameters for video path, display video, and display plot
    cap = cv2.VideoCapture(vid_path)  # Open the video file
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):  # If the video capture object is not opened successfully
        print("Error opening video stream or file")  # Print an error message

    # Define Hough search space parameters
    beta = np.linspace(-np.pi/2,np.pi/2,180)  # Create an array of angles from -90 to 90 degrees
    rho_max = int(np.sqrt(1920**2 + 1080**2))  # Maximum distance from origin to the top-right corner of the frame
    rho = np.linspace(0,rho_max,800)  # Divide the distance range into 800 equal parts
    lines = []  # Initialize an empty list to store lines in Hough space

    for i in rho:
        for j in beta:
            lines.append([i,j])  # Populate Hough search space with rho and beta combinations
        
    lines = np.array(lines)  # Convert lines to a NumPy array
 
    x_range = np.linspace(0,1920,10)  # Divide the x-axis range (width of the frame) into 10 equal parts

    num = 1  # Initialize frame count
    final_corners = []  # Initialize a list to store final corner points
    
    # Read until video is completed
    while(cap.isOpened()):  # Loop until the video capture object is open
        print(f"Frame processed = {num} / 148", end="\r", flush=True)  # Print processing progress
        num+=1  # Increment the frame count

        # Capture frame-by-frame
        ret, og_frame = cap.read()  # Read the next frame from the video
        count = np.zeros(lines.shape[0])  # Initialize an array to count votes for each line in Hough space

        if ret == True:  # If a frame is successfully read
            
            frame = cv2.GaussianBlur(og_frame,(5,5),cv2.BORDER_DEFAULT)  # Apply Gaussian blur to the frame to reduce noise
            
            edges = cv2.Canny(frame,150,350)  # Detect edges in the frame using the Canny edge detection algorithm
            pts = np.argwhere(edges != 0)  # Find the coordinates of edge points in the frame

            # Count votes for each line in Hough space
            for pt in pts:
                res = np.round((pt[1]*np.cos(lines[:,1]) + pt[0]*np.sin(lines[:,1])) - lines[:,0]).astype(np.int32)
                count[res == 0] += 1
            
            top = np.sort(count.flatten())[-20:][::-1]  # Select the top 20 counts (votes) for lines
            ind = np.argwhere(count == top[0])[0]  # Find the index of the line with the highest count (vote)
            hypothesis = lines[ind,:][0]  # Get the parameters of the line with the highest count
            best = [hypothesis]  # Initialize a list to store the best lines

            # Select the best lines (corners)
            for t in top:
                ind = np.argwhere(count == t)[0]  # Find the index of the line with the current count
                hypothesis = lines[ind,:][0]  # Get the parameters of the line with the current count
                
                # Perform a duplicate check to avoid selecting duplicate lines
                temp = np.array(best)
                if np.argwhere(np.abs(temp[:,0] - hypothesis[0]) < 140).size == 0:
                    best.append(hypothesis)  # Add the line to the list of best lines if it's not a duplicate
                
                if len(best) == 4:  # Break the loop if four corners are found
                    break
            
            best = np.array(best)  # Convert the list of best lines to a NumPy array
            
            # Find corner points using the best lines
            corners = set()
            for line1 in best:
                for line2 in best:
                    if np.array_equal(line1,line2):
                        continue
                    x = (line1[0]/np.sin(line1[1]) - line2[0]/np.sin(line2[1]))*(1/(1/np.tan(line1[1]) - 1/np.tan(line2[1])))
                    y = (line1[0] - x*np.cos(line1[1]))/np.sin(line1[1])
                    if not(0<x<1920) or not(0<y<1080):
                        continue
                    corners.add((y.round(3),x.round(3)))
                    # corners.append([y,x])

            corners = list(corners)
            corners = np.array(corners)

            # Check if the number of corners detected is 4
            if corners.shape[0] == 4:
                # Find the leftmost corner
                lc = corners[np.argwhere(corners[:,1] == np.min(corners[:,1])), :]
                # Find the rightmost corner
                rc = corners[np.argwhere(corners[:,1] == np.max(corners[:,1])), :]
                # Find the top corner
                tc = corners[np.argwhere(corners[:,0] == np.min(corners[:,0])), :]
                # Find the bottom corner
                bc = corners[np.argwhere(corners[:,0] == np.max(corners[:,0])), :]

                # Append the corners to the final list
                final_corners.append(np.array([lc,tc,rc,bc]))

            # Plotting
            if disp_plot:
                fig, ax = plt.subplots()
                im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for line in best:
                    ax.plot(x_range, (line[0] - x_range*np.cos(line[1]))/(np.sin(line[1])), color='red')
                ax.scatter(corners[:,1],corners[:,0], s=5, color='red')
                ax.set_xlabel('X coordinates')
                ax.set_ylabel('Y coordinates')
                ax.set_xlim(0, 1920)
                ax.set_ylim(0, 1080)
                ax.invert_yaxis()
                    
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                #plt.show()

                # Press Q on keyboard to  exit
            if disp_vid:
                cv2.imshow('Edges', edges)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

    # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

    return final_corners

def getH(final_corners, K):
    R_bank = []
    T_bank = []

    # Detected points in world coordinate system
    P = np.array([[0,0,1],[0,0.279,1],[0.216,0.279,1],[0.216,0,1]])

    for i in range(len(final_corners)):
        p = final_corners[i].reshape((4,2))
        A = np.zeros((p.shape[0]*2,9))

        for i in range(p.shape[0]):
            x = P[i,0]
            x_ = p[i,1]
            y = P[i,1]
            y_ = p[i,0]
            A[i*2:i*2 + 2,:] = np.array([[-x, -y, -1, 0, 0, 0, x*x_, y*x_, x_],  # Fill matrix A with coordinates
                                          [0, 0, 0, -x, -y, -1, x*y_, y*y_, y_]])

        # SVD solution
        _ , _ , V = np.linalg.svd(A)  # Singular Value Decomposition
        h = V.T[:,8]  # Extract the last column of V
        H = np.reshape(h, (3,3))  # Reshape the vector into a 3x3 matrix

        E = np.linalg.inv(K) @ H  # Estimate Essential Matrix from camera matrix and homography matrix

        x_unit = E[:,0]/np.linalg.norm(E[:,0])  # Extract x-axis unit vector
        y_unit = E[:,1]/np.linalg.norm(E[:,1])  # Extract y-axis unit vector
        scale = 2/(np.linalg.norm(E[:,0]) + np.linalg.norm(E[:,1]))  # Compute scale factor
        T = E[:,-1] * scale  # Extract translation vector

        z_unit = np.cross(x_unit,y_unit)  # Compute z-axis unit vector as cross product of x and y unit vectors

        R = np.vstack((x_unit, y_unit, z_unit)).T  # Construct rotation matrix

        r =  Rotation.from_matrix(R)  # Create a rotation object from the rotation matrix
        angles = r.as_euler("xyz",degrees=True)  # Extract Euler angles in degrees

        R_bank.append(angles)  # Append Euler angles to rotation bank
        T_bank.append(T)  # Append translation vector to translation bank

    return R_bank, T_bank  # Return rotation and translation banks



def main():
    vid_path = "main_video.avi"  # Path to the video file
    f_corners = getTrack(vid_path=vid_path, disp_vid=False, disp_plot=False)  # Detect corners in the video frames
    #K = np.array([[1380,0,946],[0,1380,527],[0,0,1]])  # Intrinsic camera matrix (optional)
    K = np.array([[767.22348963,0,759.3332939],[0,755.79620519,380.08386295],[0,0,1]])  # Intrinsic camera matrix
    R_bank, T_bank = getH(final_corners=f_corners, K = K)  # Estimate rotation and translation
    R_bank = np.array(R_bank)  # Convert rotation bank to numpy array
    T_bank = np.array(T_bank)  # Convert translation bank to numpy array

    """Plotting the translation and rotation of the camera's coordinate frame against the frame number"""

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))  # Create a figure with two subplots
    x = range(T_bank.shape[0])  # Generate frame numbers

    # Plot the first graph in the first subplot
    y1 = R_bank[:,0]  # Extract roll values
    y2 = R_bank[:,1]  # Extract pitch values
    y3 = R_bank[:,2]  # Extract yaw values

    # Plot the three graphs in the same plot
    axs[0].plot(x, y1, label='Roll')  # Plot roll
    axs[0].plot(x, y2, label='Pitch')  # Plot pitch
    axs[0].plot(x, y3, label='Yaw')  # Plot yaw

    # Add labels and title
    axs[0].set_xlabel('Frame #')  # Set x-axis label
    axs[0].set_ylabel('Value in degrees')  # Set y-axis label
    axs[0].set_title('Rotation change in the camera coordinates')  # Set title
    axs[0].legend()  # Add legend

    # Plot the second graph in the second subplot
    y1 = T_bank[:,0]  # Extract translation along x-axis
    y2 = T_bank[:,1]  # Extract translation along y-axis
    y3 = T_bank[:,2]  # Extract translation along z-axis

    # Plot the three graphs in the same plot
    axs[1].plot(x, y1, label='T_x')  # Plot translation along x-axis
    axs[1].plot(x, y2, label='T_y')  # Plot translation along y-axis
    axs[1].plot(x, y3, label='T_z')  # Plot translation along z-axis

    # Add labels and title
    axs[1].set_xlabel('Frame #')  # Set x-axis label
    axs[1].set_ylabel('Value in metres')  # Set y-axis label
    axs[1].set_title('Translation change in the camera coordinates')  # Set title

    axs[1].legend()  # Add legend

    # Add labels and a title for the entire figure
    fig.suptitle('Camera position and rotation changes')  # Set title for the entire figure
    plt.show()  # Display the plot
    fig.savefig('results/hough.png', dpi=300)  # Save the figure as an image file

if __name__ == "__main__":
    main()  # Call the main function


