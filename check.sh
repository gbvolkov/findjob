echo "ERRORS:"
cat nohup.out | grep ERROR
echo ""
echo "PROCESS:"
ps aux | grep findjob_bot.py