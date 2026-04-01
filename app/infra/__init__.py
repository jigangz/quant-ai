"""
Infrastructure abstractions.

All cloud services are accessed through abstract interfaces.
Swap implementations via environment config without changing business logic.

Broker (Kafka/Kinesis/Redis):
    from app.infra.broker import get_broker

Queue (SQS/Redis RQ/Celery):
    from app.infra.queue import get_queue

Notification (SNS/Redis Pub-Sub/webhook):
    from app.infra.notify import get_notifier

Serverless (Lambda/local function):
    from app.infra.functions import get_function_runner
"""
