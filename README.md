# stocks
## Scripts to run stock reports on demand.
### Overview:
##### Sent9.py - Scrapes stock prices and creates visual; it saves visual as Figure1.png.
##### send_email.py - Reads GMAIL credentials and uses GMAIL api to email Figure1.png.
##### Dockerfile - If you are cool with Docker, then have at it...
### Setup:
This solution uses GMAIL for emailing results, as the actual visual may not be presentable, if you are running in Docker or from a remote terminal. 

The changes are simple.
1. Acquire GMAIL auth password for application.
2. Create a new file, named secret.key and simply paste the GMAIL auth password into it. The secret.key file should be in the same directory as the scripts.
3. Update ssmtp.conf with the GMAIL auth password.The field to update is AuthPass.
4. Please update send_email.py with YOUR email address, as it defaults to mine. The fields to update are: fromaddr and toaddr.

You are all set at this point.
### Usage.
Can simply run alone, in which the stocks will default to GOOG and MANH. To get personal STOCK report, simply add the stock symbols (two) on the command line following the command, "python Sent9.py". An example would appear as: python Sent9.py AAPL EBAY.
