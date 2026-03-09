async function send(){

let msg = document.getElementById("msg").value

let res = await fetch("/api/chat",{
method:"POST",
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
sessionId:"abc123",
message:msg
})
})

let data = await res.json()

let chat = document.getElementById("chat")

chat.innerHTML += "<p><b>You:</b> "+msg+"</p>"
chat.innerHTML += "<p><b>AI:</b> "+data.reply+"</p>"

document.getElementById("msg").value=""

}