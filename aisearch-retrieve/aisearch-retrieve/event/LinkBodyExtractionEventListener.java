package com.linkpoket.source.domain.aisearch.event;

import java.net.InetAddress;
import java.net.URI;
import java.security.MessageDigest;
import java.util.Optional;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

import com.linkpoket.source.domain.aisearch.entity.AiItemBody;
import com.linkpoket.source.domain.aisearch.entity.ExtractionStatus;
import com.linkpoket.source.domain.aisearch.repository.AiItemBodyDataAccess;
import com.linkpoket.source.global.util.Snowflake;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 링크 본문 텍스트 추출 이벤트 리스너입니다.
 * 링크 저장/수정 트랜잭션 커밋 이후 비동기로 실행되며, Jsoup으로 URL을 방문해
 * 본문 텍스트를 추출한 뒤 ai_item_body 테이블에 저장합니다.
 *
 * SSRF 방어는 UrlPreview 도메인의 OpenGraphStrategy와 동일한 방식(사설/루프백 대역 차단)을 따릅니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class LinkBodyExtractionEventListener {

	private static final String ITEM_TYPE_LINK = "LINK";
	private static final int TIMEOUT_MS = 8000;
	private static final int MAX_STORED_CHARS = 20000;

	private final AiItemBodyDataAccess itemBodyDataAccess;
	private final Snowflake snowflake;

	/**
	 * 링크 본문 추출 요청을 비동기로 처리합니다.
	 * 실패해도 링크 저장 자체에는 영향이 없도록 예외를 밖으로 던지지 않습니다.
	 *
	 * @param event 본문 추출 요청 이벤트
	 */
	@TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
	@Async("threadPoolTaskExecutor")
	@Transactional(propagation = Propagation.REQUIRES_NEW)
	public void handleLinkBodyExtraction(final LinkBodyExtractionEvent event) {
		final Optional<AiItemBody> existing = itemBodyDataAccess.findByItemTypeAndItemId(ITEM_TYPE_LINK, event.linkId());
		final AiItemBody body = existing.orElseGet(() -> AiItemBody.builder()
			.id(snowflake.nextId())
			.itemType(ITEM_TYPE_LINK)
			.itemId(event.linkId())
			.extractionStatus(ExtractionStatus.PENDING)
			.build());

		try {
			validateSsrf(event.url());

			final Document doc = Jsoup.connect(event.url())
				.userAgent("Mozilla/5.0")
				.timeout(TIMEOUT_MS)
				.get();

			// 광고·네비게이션·스크립트 등 본문과 무관한 요소 제거 후 본문만 추출
			doc.select("script, style, nav, header, footer, noscript, iframe, form").remove();
			final String rawText = doc.body() != null ? doc.body().text() : doc.text();

			if (rawText == null || rawText.isBlank()) {
				body.markAsFailed("본문 텍스트를 추출할 수 없음(빈 페이지 또는 JS 렌더링 필요)");
				itemBodyDataAccess.save(body);
				log.info("[AiItemBody] extraction empty. linkId={}, url={}", event.linkId(), event.url());
				return;
			}

			final String truncated = rawText.length() > MAX_STORED_CHARS
				? rawText.substring(0, MAX_STORED_CHARS) : rawText;
			body.markAsDone(truncated, sha256(truncated));
			itemBodyDataAccess.save(body);
			log.info("[AiItemBody] extraction done. linkId={}, chars={}", event.linkId(), truncated.length());
		} catch (final Exception e) {
			body.markAsFailed(abbreviate(e.getMessage()));
			itemBodyDataAccess.save(body);
			log.warn("[AiItemBody] extraction failed. linkId={}, url={}, cause={}",
				event.linkId(), event.url(), e.toString());
		}
	}

	/**
	 * URL의 실제 IP가 내부 네트워크 대역에 속하는지 검사합니다.
	 * UrlPreview 도메인의 OpenGraphStrategy와 동일한 정책(루프백/링크로컬/사이트로컬 차단)입니다.
	 *
	 * @param url 검사할 URL
	 * @throws IllegalArgumentException 내부 네트워크 대역 IP로 판별된 경우
	 */
	private void validateSsrf(final String url) {
		try {
			final URI uri = URI.create(url);
			final InetAddress address = InetAddress.getByName(uri.getHost());
			if (address.isLoopbackAddress()
				|| address.isLinkLocalAddress()
				|| address.isSiteLocalAddress()
				|| address.isAnyLocalAddress()) {
				throw new IllegalArgumentException("내부 네트워크 대역 IP는 추출 대상에서 제외됨: " + address.getHostAddress());
			}
		} catch (final IllegalArgumentException e) {
			throw e;
		} catch (final Exception e) {
			throw new IllegalArgumentException("SSRF 검증 실패: " + e.getMessage(), e);
		}
	}

	private String sha256(final String text) {
		try {
			final MessageDigest digest = MessageDigest.getInstance("SHA-256");
			final byte[] hash = digest.digest(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
			final StringBuilder hex = new StringBuilder();
			for (final byte b : hash) {
				hex.append(String.format("%02x", b));
			}
			return hex.toString();
		} catch (final Exception e) {
			return "";
		}
	}

	private String abbreviate(final String text) {
		if (text == null) {
			return "unknown error";
		}
		return text.length() <= 255 ? text : text.substring(0, 255);
	}
}
