document.addEventListener("DOMContentLoaded", function () {
    // Function to start the camera feed separately for both cameras
    async function startCameraFeed(videoElementId) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });

            // Assign the stream to the specified video element
            const videoElement = document.getElementById(videoElementId);
            if (videoElement) {
                videoElement.srcObject = stream;
            } else {
                console.error(`Camera element with ID '${videoElementId}' not found!`);
            }
        } catch (err) {
            console.error("Error accessing the camera: ", err);
            alert("Camera access denied or unavailable. Please check browser permissions.");
        }
    }

    // Start the camera feed for each video element
    startCameraFeed('truckCamera');
    startCameraFeed('plateCamera');

    // Function to show the chart
    function showChart() {
        const chartContainer = document.getElementById('chartContainer');
        chartContainer.style.display = 'block';

        const ctx = document.getElementById('myChart').getContext('2d');

        // Fetching data from the sidebar
        const passedTrucks = parseInt(document.getElementById('passedTrucks').textContent);
        const totalTrucks = parseInt(document.getElementById('totalTrucks').textContent);
        const rejectedTrucks = parseInt(document.getElementById('rejectedTrucks').textContent);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Passed Trucks', 'Total Trucks', 'Rejected Trucks'],
                datasets: [{
                    label: 'Truck Statistics',
                    data: [passedTrucks, totalTrucks, rejectedTrucks],
                    backgroundColor: [
                        'rgba(79, 54, 4, 0.88)', // Passed Trucks
                        'rgba(19, 137, 184, 0.78)', // Total Trucks
                        'rgba(7, 130, 25, 0.78)'  // Rejected Trucks
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Event listener for the Analysis button
    document.getElementById('analysisBtn').addEventListener('click', showChart);
    document.getElementById('analysisBtnSidebar').addEventListener('click', showChart); // Sidebar button

    // Function to send AJAX request
    function updateTruckData(status) {
        const passedTrucks = parseInt(document.getElementById('passedTrucks').textContent);
        const totalTrucks = parseInt(document.getElementById('totalTrucks').textContent);
        const rejectedTrucks = parseInt(document.getElementById('rejectedTrucks').textContent);

        fetch('update_trucks.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                status: status,
                trucks_passed: passedTrucks,
                total_trucks: totalTrucks,
                rejected_trucks: rejectedTrucks
            })
        })
        .then(response => response.text())
        .then(data => alert('Truck data stored successfully!'))
        .catch(error => alert('Failed to store truck data.'));
    }

    // Event listeners for the buttons
    document.getElementById('handlePassedTrucks').addEventListener('click', function() {
        updateTruckData('passed');
    });
    document.getElementById('handleRejectedTrucks').addEventListener('click', function() {
        updateTruckData('rejected');
    });
});
