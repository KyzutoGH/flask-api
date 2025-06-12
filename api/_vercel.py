from api.index import app as application

# Handler untuk Vercel
def handler(request):
    with application.app_context():
        response = application.full_dispatch_request(request)
        return response