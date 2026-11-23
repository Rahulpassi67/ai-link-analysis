"""Safe, structured service errors used by the API boundary."""


class ServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 502
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status_code = status_code


class WebsiteExtractionError(ServiceError):
    pass


class ExternalServiceError(ServiceError):
    pass


class WebsiteContentError(ServiceError):
    def __init__(self, message: str) -> None:
        super().__init__("insufficient_website_content", message, 422)


def classify_external_error(service: str, error: Exception) -> ExternalServiceError:
    """Convert provider errors to safe public messages without echoing raw text."""
    error_name = type(error).__name__.casefold()
    detail = str(error).casefold()

    if isinstance(error, TimeoutError) or "timeout" in error_name or "deadline_exceeded" in detail:
        return ExternalServiceError(
            f"{service.casefold()}_timeout",
            f"{service} did not respond before the request timed out.",
            504
        )

    if any(marker in detail for marker in (
        "429", "resource_exhausted", "rate limit", "quota", "too many requests"
    )):
        return ExternalServiceError(
            f"{service.casefold()}_rate_limited",
            f"{service} is temporarily rate limited. Please try again later.",
            503
        )

    if any(marker in detail for marker in ("401", "403", "unauthorized", "api key")):
        return ExternalServiceError(
            f"{service.casefold()}_authentication_failed",
            f"{service} rejected its configured credentials. Check the provider configuration.",
            503
        )

    return ExternalServiceError(
        f"{service.casefold()}_unavailable",
        f"{service} could not complete the request. Please try again later.",
        502
    )
