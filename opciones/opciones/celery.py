from __future__ import absolute_import, unicode_literals
import os

from celery import Celery
from django.conf import settings

# se registra el celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'opciones.settings')
app = Celery('opciones')

app.conf.enable_utc = False
app.conf.update(timezone = 'America/Santiago')

#app.config_from_object(settings, namespace='CELERY')
app.config_from_object('django.conf:settings', namespace='CELERY')

app.conf.beat_schedule = {
    'every-10-seconds' : {
        'task': 'seleccion.tasks.update_stock',
        'schedule': 10,
    },
}

app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')