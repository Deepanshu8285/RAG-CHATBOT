css = '''
<style>
.chat-message {
    padding: 1.2rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 15%;
}
.chat-message .avatar img {
    max-width: 48px;
    max-height: 48px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 0 1rem;
    color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://api.dicebear.com/7.x/bottts/svg?seed=Bot" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://api.dicebear.com/7.x/personas/svg?seed=You">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''